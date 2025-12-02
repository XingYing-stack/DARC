#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import stopit  # 1. Import the thread-safe 'stopit' library

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument('--max_model_len', type=int, default=8192,
                    help='Maximum model sequence length (tokens) for the vLLM engine. Reduce to save KV cache memory.')
parser.add_argument('--n_candidates', type=int, default=int(os.getenv('VLLM_N_CANDIDATES', '10')),
                    help='Number of candidate answers to sample per question.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    max_model_len=args.max_model_len,
)

# Allow overriding solver generation behavior via env to tune difficulty pressure
solver_temp = float(os.getenv('SOLVER_TEMPERATURE', '1.0'))
solver_top_p = float(os.getenv('SOLVER_TOP_P', '1.0'))
solver_top_k = int(os.getenv('SOLVER_TOP_K', '40'))
solver_max_tokens = int(os.getenv('SOLVER_MAX_TOKENS', '4096'))

sample_params = vllm.SamplingParams(
    max_tokens=solver_max_tokens,
    temperature=solver_temp,
    top_p=solver_top_p,
    top_k=solver_top_k,
    stop_token_ids=[tokenizer.eos_token_id],
    n=args.n_candidates, # default 10; reduce via --n_candidates to mitigate timeouts
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)
    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Difficulty-Aware: success-rate vs golden) ----------
    def process_single(question, golden_answer, response):
        '''
        Consolidates vLLM outputs for a single question and returns a success rate
        w.r.t. the provided golden_answer. Unlike the legacy flow, we do NOT gate the
        score on majority-vote equality; instead we compute the fraction of samples
        equivalent to the golden answer using mathruler with timeouts.
        '''
        # Extract boxed content from all candidate outputs
        results = [extract_boxed_content(out.text) for out in response.outputs]
        results = [r for r in results if r]

        # Keep majority computation only for debugging/telemetry
        answer_counts = {}
        for r in results:
            if not r:
                continue
            matched_to_existing = False
            for exist in list(answer_counts.keys()):
                # Cheap string check before expensive calls
                if r == exist or ('no ' in r.lower() and 'no ' in exist.lower()):
                    answer_counts[exist] += 1
                    matched_to_existing = True
                    break
                try:
                    is_match = False
                    m1 = grade_answer_with_timeout(r, exist, timeout=10)
                    if m1 == 'TIMED_OUT':
                        print(f"      [cluster] TIMEOUT '{r[:30]}...' vs '{exist[:30]}...'")
                    elif m1:
                        is_match = True
                    if not is_match:
                        m2 = grade_answer_with_timeout(exist, r, timeout=10)
                        if m2 == 'TIMED_OUT':
                            print(f"      [cluster] TIMEOUT '{exist[:30]}...' vs '{r[:30]}...'")
                        elif m2:
                            is_match = True
                    if is_match:
                        answer_counts[exist] += 1
                        matched_to_existing = True
                        break
                except Exception as e:
                    print(f"      [cluster] ERROR comparing '{r[:30]}...' and '{exist[:30]}...': {e}")
            if not matched_to_existing:
                answer_counts[r] = 1

        if answer_counts:
            majority_ans = max(answer_counts, key=answer_counts.get)
        else:
            majority_ans = ''

        # Compute success rate against golden_answer (the only score used downstream)
        matches = 0
        total = len(results)
        for r in results:
            if not r:
                continue
            matched = False
            # Cheap equality first
            if r == golden_answer or ('no ' in r.lower() and 'no ' in golden_answer.lower()):
                matched = True
            if not matched:
                try:
                    m1 = grade_answer_with_timeout(r, golden_answer, timeout=10)
                    if m1 == 'TIMED_OUT':
                        # Try reverse direction when timed out or False
                        m2 = grade_answer_with_timeout(golden_answer, r, timeout=10)
                        matched = (m2 != 'TIMED_OUT') and bool(m2)
                    else:
                        matched = bool(m1)
                    if not matched:
                        # Try reverse direction if the first check was False
                        m2 = grade_answer_with_timeout(golden_answer, r, timeout=10)
                        matched = (m2 != 'TIMED_OUT') and bool(m2)
                except Exception as e:
                    print(f"      [grader] ERROR comparing to golden: '{r[:30]}...' vs '{golden_answer[:30]}...': {e}")
            if matched:
                matches += 1

        success_rate = (matches / total) if total > 0 else 0.0

        return {
            'question': question,
            'answer':   majority_ans,  # keep for debugging
            'score':    success_rate,  # key change: return success rate w.r.t. golden
            'results':  results,
        }

    results_all = []
    response_idx = 0
    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except Exception as e:
            # Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

@app.route('/related', methods=['GET'])
def related():
    '''Judge if question is related to text using the same solver model.

    Input file format (JSON list):
    [
      {"text": "...", "question": "..."},
      ...
    ]

    Writes a results file with entries:
    {"text": "...", "question": "...", "related": true/false, "score": 1 or 0}
    '''

    # Pause idle worker during generation
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f"[server][related] Received request for task file: {name}")

    # Load data
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    # Prepare chats
    chats = []
    items = []
    for item in data:
        text = item.get('text', '') or ''
        question = item.get('question', '') or ''
        if not text or not question:
            items.append({"text": text, "question": question, "related": False, "score": 0})
            continue
        user_content = (
            "Do you think this question `" + question + "` is related to the following text:"\
            + text + "\n\nReturn with yes or no, no other text."
        )
        chats.append([
            {"role": "system", "content": "You are a precise judge. Reply with yes or no only."},
            {"role": "user", "content": user_content},
        ])
        items.append({"text": text, "question": question})

    # Build prompts
    if chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in chats
            ]
    else:
        prompts = []

    # Generation params for quick yes/no
    sampling_params_related = vllm.SamplingParams(
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        stop_token_ids=[tokenizer.eos_token_id],
        n=1,
    )

    # Generate
    if prompts:
        responses = model.generate(prompts, sampling_params=sampling_params_related, use_tqdm=False)
    else:
        responses = []

    # Parse outputs
    out_items = []
    r_idx = 0
    for base in items:
        if 'related' in base:
            out_items.append(base)
            continue
        resp = responses[r_idx]
        r_idx += 1
        text_out = resp.outputs[0].text.strip() if resp.outputs else ''
        low = text_out.lower()
        # robust yes/no parse
        if 'yes' in low and 'no' not in low:
            related_flag = True
        elif 'no' in low and 'yes' not in low:
            related_flag = False
        else:
            # fallback: prefix token
            related_flag = low.startswith('y') and not low.startswith('n')
        out_items.append({
            "text": base.get('text', ''),
            "question": base.get('question', ''),
            "related": bool(related_flag),
            "score": 1 if related_flag else 0,
            "raw": text_out,
        })

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(out_items, f, indent=2)

    pause_event.clear()
    print(f"[server][related] Processed {name}, results saved to {out_path}. Resuming idle worker.")
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
