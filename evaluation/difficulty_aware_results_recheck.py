import json
from openai import OpenAI
from tqdm import tqdm
import random
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed




# python evaluation/difficulty_aware_results_recheck.py --result_path /share_data/data1/fanshengda/DEvo/ckpts/evaluation/_share_data_data1_fanshengda_DEvo_ckpts_models_qwen3-4b-difficulty_aware_solver_1214_global_step_64_actor_huggingface_
parser = argparse.ArgumentParser()
parser.add_argument("--result_path", type=str)
parser.add_argument("--model_name", type=str, default="gpt-4o")
args = parser.parse_args()

# ----------------------------
#   保持硬编码不动
# ----------------------------
api_urls = ["https://api.gptsapi.net/v1"]
api_keys = ["sk-54oeef36602430a9a264f8ea73ec8365c3d39c5622d24XMb"]

# 提前建好 client 列表（每个 url/key 一个 client）
clients = [
    OpenAI(base_url=api_url, api_key=api_key)
    for api_url, api_key in zip(api_urls, api_keys)
]


# ----------------------------
#   带重试的请求函数（用 OpenAI SDK + base_url）
# ----------------------------
def call_api_with_retry(answer, response, max_retries=5):
    delay = 1
    for attempt in range(max_retries):
        try:
            # 随机选一个后端（虽然现在就一个，先按你原来的写法保留）
            api_index = random.randint(0, len(clients) - 1)
            client = clients[api_index]

            completion = client.chat.completions.create(
                # 保持你原来 payload 里的 hardcode，不用 args.model_name
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a math answer checker."},
                    {
                        "role": "user",
                        "content": (
                            f"Hi, there is a answer: {answer}\n\n, "
                            f"and the ground truth answer is: {response}\n\n, "
                            f"please check whether the answer is correct or not, "
                            f"and return the **only** Yes or No."
                        ),
                    },
                ],
                temperature=0.1,
                timeout=20,
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"[Retry {attempt+1}/{max_retries}] Error: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # 指数退避
            else:
                return "No"

    return "No"


# ----------------------------
#   单样本包装函数
# ----------------------------
def process_example_parallel(item):
    answer, response = item["answer"], item["response"]
    score = item["score"]
    idx = item["idx"]  # 保留索引用于外面放回

    if score >= 0.5:
        return idx, score  # 不需要调用 API

    gpt_check = call_api_with_retry(answer, response)
    if "yes" in gpt_check.lower():
        score = 1

    return idx, score


# ----------------------------
#   主处理逻辑（并发）
# ----------------------------
new_results = []

for model_name in [args.model_name]:
    for dataset in ["math", "gsm8k", "amc", "minerva", "olympiad", "aime2024", "aime2025"]:

        with open(f'{args.result_path}/results_{dataset}.json', 'r') as f:
            results = json.load(f)

        # 和你原来一样：len(results)-1，最后一个不参与打分
        items = [
            {
                "idx": i,
                "answer": results[i]['answer'],
                "response": results[i]['response'],
                "score": results[i]['score']
            }
            for i in range(len(results) - 1)
        ]

        # 并发执行
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_example_parallel, item) for item in items]

            for future in tqdm(as_completed(futures), total=len(items), desc=f"Processing {dataset}"):
                idx, new_score = future.result()
                results[idx]['score'] = new_score

        avg_score = round(sum(r['score'] for r in results[:-1]) / len(results[:-1]) * 100, 2)

        new_results.append({
            "model": model_name,
            "dataset": dataset,
            "score": avg_score
        })

        print(new_results)

        with open(f'{args.result_path}/final_results.json', "a") as f:
            json.dump({"model": model_name, "dataset": dataset, "score": avg_score}, f)
            f.write("\n")