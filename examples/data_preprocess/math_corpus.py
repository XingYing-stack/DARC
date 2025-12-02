# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os
import random
import re
import datasets
import pandas as pd
from datasets import Dataset

CHALLENGER_PROMPT_TEMPLATE = '''Your task is to generate a single self-contained question and its correct answer inspired by the given document.
The question must strictly satisfy both the difficulty_id and the answer_type constraints.

You must output exactly one JSON object as specified below.  
All reasoning MUST be placed inside the "analysis" field of the JSON.

## Document
[BEGIN]
{text}
[END]

## Difficulty Level

You are given a target difficulty level:

- difficulty_id: {difficulty_id}

Difficulty levels follow these operational definitions:

1 — Very Easy:
  - Uses one short span of information
  - Exactly 1 reasoning step
  - Direct fact recall with minimal inference

2 — Easy:
  - Uses 1–2 spans
  - 1–2 reasoning steps
  - Simple inference or transformation

3 — Moderate:
  - Uses at least 2 distinct spans
  - 2–3 reasoning steps
  - Requires combining multiple facts

4 — Hard:
  - Uses 2–3 distant spans
  - 3–5 reasoning steps
  - Requires conditional reasoning or multi-variable inference

5 — Very Hard:
  - Uses 3+ distant spans across the document
  - 5+ reasoning steps
  - Requires global synthesis, constrained reasoning, or multi-stage calculations

Your generated question and solution process must match the target difficulty_id as closely as possible.

## Answer Type

You must generate a question whose answer has the following type:

- answer_type: {answer_type}

Rules:
- integer → JSON integer
- float → JSON number with a decimal point

## Core Requirements for the Question

1. The question must be inspired by the document (but self-contained).
2. The question must not reference “the document” or “the text”.
3. The question must be understandable by someone who only sees the question.
4. The reasoning steps must match the difficulty_id.
5. The question must combine the number of spans required by difficulty_id.
6. The answer must be unique and consistent with the document.
7. All variables must be defined in the question itself.
8. No ambiguity.

---

# Final Output Format (STRICT)

Your output must be **exactly one JSON object** with the following fields:

- "analysis" (string)
- "question" (string)
- "intermediate_results" (object)
- "answer" (number)
- "solving_time_estimate" (number)
- "required_concepts" (array of strings)
- "potential_errors" (array of strings)

## Field Specifications

### 1. "analysis" (string)
This field contains your full internal reasoning:
- selection of spans
- mapping to difficulty
- design of question
- intermediate calculations
- validation of uniqueness
- check for answer_type

This is the ONLY place where long reasoning is allowed.

### 2. "question" (string)
- A single natural-language exam-style question.
- Completely self-contained.
- No reasoning, no hints, no metadata.

### 3. "intermediate_results" (object)
- Keys: short step names
- Values: 1–20 sentence summaries of key reasoning steps

### 4. "answer" (number)
- Must be numeric
- Must satisfy the answer_type

### 5. "solving_time_estimate" (number)
Minutes required.

### 6. "required_concepts" (array of strings)
1–10 relevant concepts from document.

### 7. "potential_errors" (array of strings)
1–10 potential student mistakes.

---

# Example Output (do not copy; only follow structure)

```json
{{
  "analysis": "I will design a self-contained calculus problem using a simple polynomial. To ensure the answer is an integer and matches the required type, I select f(x) = 2x and integrate over a defined interval. I check that the computation requires a couple of steps but remains easy. I also verify uniqueness and the numeric type.",
  "question": "Let f(x) = 2x for x in [1, 3]. Define F(x) as the integral from 1 to x of 2t dt. What is the value of F(3)?",
  "intermediate_results": {{
    "step1_plan_question": "Chose an integrand ensuring an integer final result.",
    "step2_compute_integral": "Compute ∫ from 1 to 3 of 2t dt = [t^2] from 1 to 3 = 9 - 1 = 8.",
    "step3_validate_type": "Result is integer, matching requirements."
  }},
  "answer": 8,
  "solving_time_estimate": 4,
  "required_concepts": [
    "Definite integral",
    "Antiderivative",
    "Polynomial integration",
    "Fundamental Theorem of Calculus"
  ],
  "potential_errors": [
    "Incorrect evaluation of bounds",
    "Arithmetic mistake in 9 - 1",
    "Confusing F(3) with F(3) - F(1)"
  ]
}}
```
'''


def _gen_single_item(text: str, idx: int, difficulty_id: int, answer_type: str):
    """构造一条样本（给定 text/doc 与目标难度/答案类型）。"""
    question = CHALLENGER_PROMPT_TEMPLATE.format(
        difficulty_id=difficulty_id,
        answer_type=answer_type,
        text=text,
    )

    return {
        "data_source": "questioner_given_difficulty_id",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": difficulty_id},
        "extra_info": {
            "split": "train",
            "index": idx,  # 原始文档 id，用于成组排序奖励
            "answer_type": answer_type,
            "text": text,
        },
    }


def process_batch_fn(batch, indices, difficulties=(1, 2, 3, 4, 5), k_per_doc=None, int_ratio=0.8):
    """
    将每条原始文档扩展为多个难度标注的样本，默认覆盖 1..5 全部难度。

    - difficulties: 可选难度集合，例如 (1,2,3,4,5)
    - k_per_doc: 若不为 None，则从 difficulties 中随机采样 k 个难度（避免数据量膨胀过大）
    - int_ratio: 生成整数答案类型的概率（其余为 float）
    """
    out = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
    }

    n = len(batch["text"])
    for i in range(n):
        text = (batch["text"][i] or "")[:15000]
        doc_idx = indices[i] if isinstance(indices, list) else indices[i]

        # 选择该文档要生成的难度集合
        if k_per_doc is not None and k_per_doc < len(difficulties):
            diffs = random.sample(list(difficulties), k_per_doc)
        else:
            diffs = list(difficulties)

        for d in sorted(diffs):
            answer_type = "integer" if random.random() < int_ratio else "float"
            item = _gen_single_item(text=text, idx=doc_idx, difficulty_id=d, answer_type=answer_type)
            for k in out.keys():
                out[k].append(item[k])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path",
        default="/share_data/data1/fanshengda/DEvo/data/part_000000.parquet",
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/share_data/data1/fanshengda/DEvo/data/challenger_1201",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--difficulties",
        default="1,2,3,4,5",
        help="Comma-separated difficulty ids to expand per document (default: all 1..5).",
    )
    parser.add_argument(
        "--k_per_doc",
        type=int,
        default=None,
        help="If set, randomly sample k difficulties per document instead of all to control dataset size.",
    )
    parser.add_argument(
        "--int_ratio",
        type=float,
        default=0.6,
        help="Probability of integer answer type; remainder uses float.",
    )

    args = parser.parse_args()

    # 1. 读原始 parquet，用 pandas -> HF Dataset
    df = pd.read_parquet(args.local_dataset_path).head(10000)
    dataset = Dataset.from_pandas(df)

    # 2. 按文档扩展多个难度版本，默认 1..5 全覆盖
    diffs = tuple(int(x) for x in re.split(r"[，,\s]+", args.difficulties.strip()) if x)
    train_dataset = dataset.map(
        lambda batch, idxs: process_batch_fn(batch, idxs, difficulties=diffs, k_per_doc=args.k_per_doc, int_ratio=args.int_ratio),
        with_indices=True,
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 3. 保存为 parquet
    os.makedirs(args.local_save_dir, exist_ok=True)
    save_path = os.path.join(args.local_save_dir, "train.parquet")
    train_dataset.to_parquet(save_path)
    print(f"Saved processed dataset to: {save_path}")
