import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
from openai import OpenAI
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from collections import Counter
import os
import re

CATEGORICAL_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer option within \\boxed{}."
    " Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
)

# _CATEGORICAL_OPTION_RE = re.compile(r"(?m)^\s*([A-J])[\.\)]\s+")
_CATEGORICAL_OPTION_RE = re.compile(
    r"(?i)(?:^|\\r\\n|\\n|\\r|\r\n|\n|\r)\s*([A-J])\s*"
    r"(?:[\.\):：\uFF0E\uFF09\u3001])\s*"
)


def categorical_question_has_options(question: str, *, min_distinct: int = 3) -> bool:
    if not isinstance(question, str) or not question.strip():
        return False
    hits = _CATEGORICAL_OPTION_RE.findall(question.upper())
    return len(set(hits)) >= int(min_distinct)


def _normalize_answer_type(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _require_answer_type(sample: dict) -> str:
    at = _normalize_answer_type(sample.get("answer_type"))
    if not at:
        raise ValueError("Missing required field `answer_type` in sample.")
    return at


def process_one(idx, messages, client: OpenAI, model_name: str):
    """
    单条调用 openai 接口。
    messages: 必须是 chat messages 列表：[{ "role": "...", "content": "..."}, ...]
    返回: (idx, accepted, decision_dict)
    """
    try:
        # 这里可以根据需要加 temperature / max_tokens 等参数
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0
        )
        content = resp.choices[0].message.content
        # 这里不解析 content，直接当作原始字符串保存
        decision = {
            "status": "ok",
            "raw": content,
        }
        accepted = True  # 如果你有过滤逻辑，可以在这里改
        return idx, accepted, decision
    except Exception as e:
        decision = {
            "status": "error",
            "raw": str(e),
        }
        return idx, False, decision


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _gen_single_item(idx, sample):
    # print(sample['question'])
    assert 'text' in sample
    assert 'answer' in sample
    assert len(sample['text']) > 0

    assert isinstance(sample, dict)
    for k in ["text", "question", "answer"]:
        assert k in sample, f"missing key: {k}"

    assert isinstance(sample["text"], str), \
        f"text must be str, got {type(sample['text'])}"

    if isinstance(sample["question"], list) or isinstance(sample["question"], tuple):
        assert len(sample["question"]) == 1
        sample["question"] = sample["question"][0]

    assert isinstance(sample["question"], str), \
        f"question must be str, got {type(sample['question'])}"

    answer_type = _require_answer_type(sample)

    if answer_type == "categorical":
        prompt = [
            {"role": "user", "content": sample["question"] + CATEGORICAL_INSTRUCTION},
        ]
        text_prompt = [
            {
                "role": "user",
                "content": (
                    "Read the following context and answer the question.\n\n"
                    f"Context:\n{sample['text']}\n\n"
                    f"Question: {sample['question']}"
                    + CATEGORICAL_INSTRUCTION
                ),
            },
        ]
    else:
        prompt = [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": sample["question"]},
        ]
        text_prompt = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {
                "role": "user",
                "content": (
                    "Read the following context and answer the question.\n\n"
                    f"Context:\n{sample['text']}\n\n"
                    f"Question: {sample['question']}"
                ),
            },
        ]

    return {
        "data_source": "solver",
        "prompt": prompt,
        "text_prompt": text_prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(sample['answer'])},
        "extra_info": {
            "split": "train",
            "index": idx,
            "text": sample['text'],
            # "analysis": str(sample['analysis']),
            'difficulty_id': sample['difficulty_id'],
            "answer_type": answer_type,
            # 'solving_time_estimate':sample['solving_time_estimate'],
        },
    }

def has_surrogate_in_anything(x) -> bool:
    # 递归检查：str / list / tuple / dict / 其它转 str
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    if isinstance(x, str):
        return any(0xD800 <= ord(c) <= 0xDFFF for c in x)
    if isinstance(x, (list, tuple)):
        return any(has_surrogate_in_anything(v) for v in x)
    if isinstance(x, dict):
        return any(has_surrogate_in_anything(k) or has_surrogate_in_anything(v) for k, v in x.items())
    # 兜底：某些对象 stringify 后可能暴露 surrogate
    try:
        s = str(x)
        return any(0xD800 <= ord(c) <= 0xDFFF for c in s)
    except Exception:
        return False


import time
import json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


SYSTEM_PROMPT = """You are a dataset filtering judge.

Decide whether to KEEP the sample.

KEEP if the TEXT clearly helps answer the QUESTION and the QUESTION can be answered without the document.

Be lenient. Do NOT over-filter.

Return ONLY one word:
true or false
"""


def _call_judge(client, model, question, text, max_retries=3):
    user_prompt = f"""QUESTION:
{question}

TEXT:
{text}

Should this sample be kept?"""

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            out = resp.choices[0].message.content.strip().lower()
            if "true" in out:
                return True
            if "false" in out:
                return False
        except Exception:
            time.sleep(0.8 * (2 ** attempt))

    # fail-open：避免误删
    return True

def assert_curriculum_order(df):
    did = df["extra_info"].apply(lambda x: x.get("difficulty_id", -1)).tolist()
    idxs = df["extra_info"].apply(lambda x: x.get("index", -1)).tolist()
    assert idxs == sorted(idxs), "extra_info.index is not non-decreasing (order corrupted?)"
    # 可选：difficulty 也应当非降（如果你的排序目标就是这个）
    assert did == sorted(did), "difficulty_id is not non-decreasing (order corrupted?)"


def filter_unrelative(df, base_url, api_key, model="gpt-5-mini", max_workers=32):
    client = OpenAI(api_key=api_key, base_url=base_url)

    data = df.to_dict(orient="records")

    questions = [sample["prompt"][1]["content"] for sample in data]
    texts = [sample["extra_info"]["text"] for sample in data]

    keep_mask = [True] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _call_judge, client, model, questions[i], texts[i]
            ): i
            for i in range(len(data))
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Filtering (LLM judge)",
        ):
            idx = futures[future]
            try:
                keep_mask[idx] = future.result()
            except Exception:
                keep_mask[idx] = True  # fail-open

    print("keep ratio:", keep_mask.count(True) / len(keep_mask))
    return df.loc[keep_mask].reset_index(drop=True)






def filter_duplicate(df, model_path):
    # =========================
    # load data
    # =========================
    data = df.to_dict(orient="records")

    questions = [
        sample["prompt"][1]["content"]
        for sample in data
    ]

    # =========================
    # load model
    # =========================
    device = "cpu"
    model = SentenceTransformer(model_path, device=device)

    # =========================
    # encode all questions
    # =========================
    embeddings = model.encode(
        questions,
        batch_size=64,
        convert_to_tensor=True,
        normalize_embeddings=True,  # VERY important
        show_progress_bar=True
    )

    # =========================
    # similarity-based filtering (reverse order)
    # =========================
    threshold = 0.6

    kept_indices = []
    kept_embeddings = []

    # ⚠️ 从后往前遍历
    for idx in tqdm(range(len(embeddings) - 1, -1, -1)):
        emb = embeddings[idx]

        if len(kept_embeddings) == 0:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
            continue

        sims = util.cos_sim(emb, torch.stack(kept_embeddings))[0]
        max_sim = sims.max().item()

        if max_sim < threshold:
            kept_indices.append(idx)
            kept_embeddings.append(emb)
        # else: filtered out (因为后面已经有代表了)

    # =========================
    # restore original order (VERY IMPORTANT)
    # =========================
    kept_indices = sorted(kept_indices)

    filtered_data = [data[i] for i in kept_indices]

    print(f"Original size: {len(data)}")
    print(f"Filtered size: {len(filtered_data)}")
    print(f"Removed: {len(data) - len(filtered_data)}")

    return pd.DataFrame(filtered_data)


# nohup python question_generate_difficulty_aware.py > ../logs/question_generate_difficulty_aware_qwen4B.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /share_data/data1/models/Qwen/Qwen3-4B-Instruct-2507 --served-model-name /share_data/data1/models/Qwen/Qwen3-4B-Instruct-2507 --max-model-len=32768 --tensor-parallel-size 4 --port 6001 --api-key dada --gpu-memory-utilization 0.9 --disable_cascade_attn
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /share_data/data1/fanshengda/DEvo/ckpts/models/qwen3-4b-difficulty_aware_questioner_1220/global_step_350/actor/huggingface --served-model-name questioner --max-model-len=32768 --tensor-parallel-size 8 --port 6000 --api-key dada --gpu-memory-utilization 0.9 --disable_cascade_attn
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="questioner")
    parser.add_argument("--api_key", type=str, default="dada")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:6000/v1")

    parser.add_argument("--judge_api_key", type=str, default="sk-kuFDU3HN9ni5EuDj6f23Ff355a0841Fb856eC63eCd27D947")
    parser.add_argument("--judge_base_url", type=str, default="https://toollearning.cn/v1")
    parser.add_argument(
        "--save_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/solver_1221/questioner_350_train.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--solver_save_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/solver_1221/solver_questioner_350_train.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--sentence_transformer_path",
        type=str,
        default="/data3/workhome/fanshengda/models/sentence-transformers/all-MiniLM-L6-v2",
        help="输出 parquet 路径",
    )

    parser.add_argument(
        "--dup_solver_save_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/solver_1221/solver_questioner_350_train_pruned.parquet",
        help="输出 parquet 路径",
    )

    parser.add_argument(
        "--parquet_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/challenger_1219/train.parquet",
        help="输入 parquet 路径",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="并发线程数",
    )

    args = parser.parse_args()

    # 确保所有输出路径的父目录存在
    ensure_parent_dir(args.save_path)
    ensure_parent_dir(args.solver_save_path)
    ensure_parent_dir(args.dup_solver_save_path)

    # 初始化 client（一般线程安全，如果不放心也可以改成每线程一个 client）
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # 读 parquet
    df = pd.read_parquet(args.parquet_path)


    # df = df.head(100)
    if "prompt" not in df.columns:
        raise ValueError(
            f"'prompt' column not found in dataframe columns: {df.columns.tolist()}"
        )

    # 准备任务列表：每个元素是 (idx, messages)
    # 其中 messages 是 openai chat 格式的 list[dict]
    tasks = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]

        # 如果 parquet 里已经是 [{'role': 'user', 'content': '...'}] 这样的列表，直接用
        if isinstance(prompt, list):
            messages = prompt
        else:
            # 否则认为是字符串，包成一条 user message
            messages = [{"role": "user", "content": str(prompt)}]

        tasks.append((idx, messages))

    results_dict = {}  # idx -> (accepted, decision)

    # 初始化结果列，避免有些 idx 没跑到时报错
    df["accepted"] = False
    df["decision"] = None

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_one, idx, messages, client, args.model
            ): idx
            for idx, messages in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Querying model",
        ):
            idx = futures[future]
            try:
                _idx, accepted, decision = future.result()
                results_dict[_idx] = (accepted, decision)
            except Exception as e:
                print(f"[ERROR] Future for idx={idx} raised exception: {e}")
                results_dict[idx] = (False, {"status": "error", "raw": None})

    # 写回到 df
    for idx, (accepted, decision) in results_dict.items():
        df.at[idx, "accepted"] = accepted
        # 保存为 JSON 字符串，方便后续解析
        df.at[idx, "decision"] = json.dumps(decision, ensure_ascii=False)

    df.to_parquet(args.save_path, index=False)
    print(f"Saved results to {args.save_path}")
    df = pd.read_parquet(args.save_path)

    df = df[df['accepted'] ==True]


    ori_solver_list = df[['reward_model', 'decision', 'extra_info']].values.tolist()

    solver_list = []

    for idx, sample in tqdm(enumerate(ori_solver_list)):
        difficulty_id = sample[0]['ground_truth']
        raw = sample[1]
        text = sample[2]['text']

        solver = json.loads(raw)
        try:
            solver = json.loads(solver['raw'])
            solver['answer_type'] = sample[2]['answer_type']
            solver['difficulty_id'] = difficulty_id
            solver['text']= text
            required_keys = {
                'analysis', 'question', 'intermediate_results', 'answer',
                'solving_time_estimate', 'required_concepts', 'potential_errors',
                'difficulty_id', 'text', 'answer_type'
            }

            if set(solver.keys()) != required_keys:
                raise KeyError

            if not isinstance(solver['solving_time_estimate'], int) and not isinstance(solver['solving_time_estimate'], float):
                solver['solving_time_estimate'] = float(solver['solving_time_estimate'])


            if not isinstance(solver['question'], str) and not isinstance(solver['question'], list) and not isinstance(solver['question'], tuple):
                raise TypeError
            solver_list.append(solver)

        except:
            print('error at index:', idx)
            continue
    solver_list = sorted(
    solver_list,
    key=lambda x: (
        x.get('difficulty_id', 0),
        len(x['intermediate_results']) if isinstance(x.get('intermediate_results'), (list, tuple)) else 100,
        x.get('solving_time_estimate', 100),
    )
)
    solver_list = [_gen_single_item(idx, sample) for idx, sample in tqdm(enumerate(solver_list))]

    solver_df = pd.DataFrame(solver_list)

    bad = solver_df[~solver_df["prompt"].apply(lambda v: isinstance(v, list))]
    print("bad rows:", len(bad))
    print(bad[["prompt"]].head(5))
    print(bad["prompt"].apply(type).value_counts())


    # 添加对于问题是否包含选项的过滤
    def _extract_user_question(prompt_cell) -> str:
        if not isinstance(prompt_cell, list):
            return ""
        for msg in prompt_cell:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _is_categorical(extra_info_cell) -> bool:
        if not isinstance(extra_info_cell, dict):
            raise TypeError("extra_info must be a dict.")
        at = extra_info_cell.get("answer_type")
        if not isinstance(at, str) or not at.strip():
            raise ValueError("extra_info.answer_type is required and must be a non-empty string.")
        return at.strip().lower() == "categorical"

    before = len(solver_df)
    drop_mask = solver_df.apply(
        lambda row: (
            _is_categorical(row.get("extra_info"))
            and (not categorical_question_has_options(_extract_user_question(row.get("prompt"))))
        ),
        axis=1,
    )
    num_drop = int(drop_mask.sum())
    if num_drop:
        solver_df = solver_df.loc[~drop_mask].reset_index(drop=True)
    print(f"Dropped {num_drop} categorical rows without options ({num_drop / max(before, 1):.4%})")

    print('Answer Type Distribution:', Counter([sample['answer_type'] for sample in solver_df['extra_info'].tolist()]))


    # ======================去除非unicode字符======================
    tqdm.pandas(desc="Scanning for invalid Unicode")

    bad_mask = solver_df.progress_apply(
        lambda row: any(has_surrogate_in_anything(v) for v in row.values),
        axis=1
    )

    num_bad = int(bad_mask.sum())
    print(f"Dropped {num_bad} rows ({num_bad / len(solver_df):.4%}) due to invalid Unicode")

    solver_df = solver_df.loc[~bad_mask].reset_index(drop=True)
    #


    solver_df.to_parquet(args.solver_save_path, index=False)
    print(f"Saved solver results to {args.solver_save_path}, len(solver_df): {len(solver_df)}")


    # solver_df = pd.read_parquet(args.solver_save_path)
    #
    # solver_df = filter_duplicate(solver_df, args.sentence_transformer_path)
    #
    # print(Counter([sample['extra_info']['difficulty_id'] for sample in solver_df.to_dict(orient="records")]))
    # assert_curriculum_order(solver_df)

    # solver_df = filter_unrelative(solver_df, base_url=args.judge_base_url, api_key=args.judge_api_key)
    # assert_curriculum_order(solver_df)

    # solver_df.to_parquet(args.dup_solver_save_path, index=False)
    # print(f"Saved solver results to {args.dup_solver_save_path}, len(pruned_solver_df): {len(solver_df)}")
