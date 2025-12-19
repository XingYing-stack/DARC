import argparse
import json
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


FILTER_PROMPT_TEMPLATE = """
You are an expert dataset auditor for a difficulty-aware question-generation system.

Your task is to determine whether the given document contains enough structured, diverse, multi-span information to support generating **ALL THREE difficulty levels**:
- Easy (1–2 spans, simple inference)
- Medium (2+ spans, moderate reasoning)
- Hard (3+ distant spans, multi-step synthesis, ≥5 reasoning steps)

A document must support **all three** difficulty levels to be considered valid.  
If any difficulty level cannot be generated due to insufficient content, the document must be rejected.

---

# Document
[BEGIN]
{text}
[END]

---

# Evaluation Criteria (STRICT)

You must evaluate whether the document contains:

### 1. Sufficient spans
- **Easy:** at least 1–2 clear factual or mathematical spans
- **Medium:** at least 2 distinct spans that relate but require combination
- **Hard:** at least 3+ *distant* spans enabling multi-step global reasoning or synthesis

### 2. Diverse reasoning types
- Quantitative, logical, relational, temporal, causal, definitional, structural, or procedural information  
- Hard questions must be realistically constructible (≥5 steps)

### 3. Answerability
- Facts are explicit, not implied or missing
- No contradictions, no hallucination traps

### 4. Multi-level reasoning potential
- Text must allow increasing difficulty by combining spans in more complex ways
- Hard difficulty must be feasible WITHOUT inventing missing info

If the document only supports Easy or Easy+Medium → reject.

---

# Output Format (STRICT)

You must output exactly **one JSON object** with these fields:

- "analysis": string  
  Detailed explanation of:
  - extracted spans  
  - whether spans can support easy / medium / hard  
  - limitations or missing information  
  - reasoning about why the document passes or fails  

- "supports_easy": boolean  
- "supports_medium": boolean  
- "supports_hard": boolean  

- "final_decision": string  
  One of:
  - "accept"  (only if all three levels are supported)
  - "reject"

- "reason_for_decision": string  
  One-sentence summary of the core reason.

- "missing_requirements": array of strings  
  List any missing elements needed for full support of all difficulty levels.

---

# Acceptance Rule

A document is accepted **ONLY IF**:
- supports_easy = true  
- supports_medium = true  
- supports_hard = true  

Otherwise:
- final_decision = "reject"
"""


def call_filter_model(client: OpenAI, model: str, text: str, max_retries: int = 3, sleep_sec: float = 2.0):
    """调用 OpenAI 模型进行文档过滤，返回解析后的 JSON。"""
    prompt = FILTER_PROMPT_TEMPLATE.format(text=text)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
            )
            content = resp.choices[0].message.content
            # 有时候模型可能外面再包一层 ```json ```，做个简单清洗
            content = content.strip()
            if content.startswith("```"):
                content = content.strip("`")
                # 去掉可能的 "json" 语言标记
                if content.lower().startswith("json"):
                    content = content[4:].strip()
            result = json.loads(content)
            return result
        except Exception as e:
            print(f"[WARN] call_filter_model error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(sleep_sec)


def process_one(idx, text, client, model):
    """单条样本的处理逻辑，用于多线程调用。"""
    result = call_filter_model(client, model, text)

    if result is None:
        # 调用失败时直接视为 reject
        accepted = False
        decision = {"status": "error", "raw": None}
        print(f"[{idx}] accepted={accepted} | ERROR in call_filter_model")
        return idx, accepted, decision

    supports_easy = bool(result.get("supports_easy", False))
    supports_medium = bool(result.get("supports_medium", False))
    supports_hard = bool(result.get("supports_hard", False))
    final_decision = result.get("final_decision", "").lower()

    accepted = (
        supports_easy and
        supports_medium and
        supports_hard and
        final_decision == "accept"
    )

    decision = {
        "status": "ok",
        "supports_easy": supports_easy,
        "supports_medium": supports_medium,
        "supports_hard": supports_hard,
        "final_decision": final_decision,
        "reason_for_decision": result.get("reason_for_decision", ""),
        "full_decision_text": json.dumps(result),
    }

    print(f"[{idx}] accepted={accepted} | easy={supports_easy}, medium={supports_medium}, hard={supports_hard}, decision={final_decision}")
    return idx, accepted, decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, default="/share_data/data1/fanshengda/DEvo/data/part_000000.parquet", help="输入 parquet 路径")
    parser.add_argument("--output_path", type=str, default="/share_data/data1/fanshengda/DEvo/data/math_filter1216.parquet", help="过滤后输出 parquet 路径")
    parser.add_argument("--api_key", type=str, default="sk-6y8kz2o3U0hG77KSQEto0s0GFWGprChx2tzO8DmL1TSfJlQ1", help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://api.moonshot.cn/v1", help="OpenAI base URL，例如 https://api.openai.com/v1 或你的自定义网关")
    parser.add_argument("--model", type=str, default="kimi-k2-0905-preview", help="用于过滤的模型名称")
    parser.add_argument("--text_column", type=str, default="text", help="存放文档内容的列名")
    parser.add_argument("--batch_limit", type=int, default=1500, help="仅调试用，限制最多处理多少行")
    parser.add_argument("--num_workers", type=int, default=32, help="并发线程数")
    args = parser.parse_args()


    # 初始化 client（多线程场景一般也是没问题的，如有问题再改成每线程一个 client）
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # 读 parquet
    if 'parquet' in args.parquet_path.lower():
        df = pd.read_parquet(args.parquet_path)
    else:
        df = pd.read_json(args.parquet_path, lines=True)
    if args.text_column not in df.columns:
        raise ValueError(f"text_column '{args.text_column}' not found in dataframe columns: {df.columns.tolist()}")

    if args.batch_limit is not None:
        # df = df.head(args.batch_limit)
        start = 32000
        end = 32000 + args.batch_limit
        df = df.iloc[start:end]

    # 准备任务列表
    tasks = []
    for idx, row in df.iterrows():
        text = str(row[args.text_column])
        text = text[:15000]
        # 截断字符串，避免爆长度
        tasks.append((idx, text))

    results_dict = {}  # idx -> (accepted, decision)

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one, idx, text, client, args.model): idx
            for idx, text in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Filtering docs"):
            idx = futures[future]
            try:
                _idx, accepted, decision = future.result()
                results_dict[_idx] = (accepted, decision)
            except Exception as e:
                print(f"[ERROR] Future for idx={idx} raised exception: {e}")
                results_dict[idx] = (False, {"status": "error", "raw": None})

    # 按原顺序组装 keep_mask / decisions
    keep_mask = []
    decisions = []
    for idx in df.index:
        accepted, decision = results_dict.get(idx, (False, {"status": "error", "raw": None}))
        keep_mask.append(accepted)
        decisions.append(decision)

    df["filter_meta"] = decisions
    df_filtered = df[pd.Series(keep_mask, index=df.index)]

    print(f"Original rows: {len(df)}, kept: {len(df_filtered)}")
    df_filtered.to_parquet(args.output_path, index=False)
    print(f"Saved filtered parquet to: {args.output_path}")


if __name__ == "__main__":
    main()
