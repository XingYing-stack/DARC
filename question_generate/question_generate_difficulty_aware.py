import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import pandas as pd
from openai import OpenAI

from examples.reward_function.difficulty_aware_questioner import difficulty_reward


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

    return {
        "data_source": "solver",
        "prompt":  [
            {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
            {"role": "user", "content": sample['question']},],
        "text_prompt": [
            {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
            {'role': 'user', 'content': "Read the following context and answer the question.\n\n"
            f"Context:\n{sample['text']}\n\n"
            f"Question: {sample['question']}"},
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": str(sample['answer'])},
        "extra_info": {
            "split": "train",
            "index": idx,
            "text": sample['text'],
            # "analysis": str(sample['analysis']),
            # 'difficulty_id': sample['difficulty_id'],
            # 'solving_time_estimate':sample['solving_time_estimate'],
        },
    }



# nohup python question_generate_difficulty_aware.py > ../logs/question_generate_difficulty_aware_qwen4B.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /share_data/data1/models/Qwen/Qwen3-4B-Instruct-2507 --served-model-name /share_data/data1/models/Qwen/Qwen3-4B-Instruct-2507 --max-model-len=32768 --tensor-parallel-size 4 --port 6001 --api-key dada --gpu-memory-utilization 0.9 --disable_cascade_attn
# CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /share_data/data1/fanshengda/DEvo/ckpts/models/qwen3-4b-difficulty_aware_questioner_1207/global_step_300/actor/huggingface --served-model-name questioner --max-model-len=32768 --tensor-parallel-size 4 --port 6000 --api-key dada --gpu-memory-utilization 0.9 --disable_cascade_attn
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="questioner")
    parser.add_argument("--api_key", type=str, default="dada")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:6000/v1")

    parser.add_argument(
        "--save_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/solver_1212/qestioner_300_train.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--solver_save_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/solver_1212/solver_questioner_300_train.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--parquet_path",
        type=str,
        default="/share_data/data1/fanshengda/DEvo/data/challenger_1212/train.parquet",
        help="输入 parquet 路径",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="并发线程数",
    )

    args = parser.parse_args()

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
    # print(f"Saved results to {args.save_path}")
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
            solver['difficulty_id'] = difficulty_id
            solver['text']= text
            required_keys = {
                'analysis', 'question', 'intermediate_results', 'answer',
                'solving_time_estimate', 'required_concepts', 'potential_errors',
                'difficulty_id', 'text'
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




    solver_df.to_parquet(args.solver_save_path, index=False)
    print(f"Saved solver results to {args.solver_save_path}, len(solver_df): {len(solver_df)}")

