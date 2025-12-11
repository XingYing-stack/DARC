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



global_text_lenghs = []


id2def = {
    1: """1 — Easy:
  - Uses 1–2 spans
  - 1–2 reasoning steps
  - Simple inference or transformation""",
    2: """2 — Moderate:
  - Uses at least 2 distinct spans
  - 2–3 reasoning steps
  - Requires combining multiple facts""",
    3: """3 — Hard:
  - Uses 3+ distant spans across the document
  - 5+ reasoning steps
  - Requires global synthesis, constrained reasoning, or multi-stage calculations"""
}

id2ICL = {
    1: """
{
  "analysis": "Construct an easy-level question requiring only one reasoning step and a single intermediate result. We use the gravitational potential energy formula U = mgh. Only one span is needed (the formula) and one arithmetic operation, fully satisfying the Easy level definition.",
  "question": "A 3 kg bowling ball is lifted to a height of 1.5 meters above the lane. Taking g = 9.81 m/s^2, what is its gravitational potential energy relative to the lane surface? Give your answer in joules, rounded to the nearest whole number.",
  "intermediate_results": {
    "step1_compute_U": "Compute U = mgh = 3 × 9.81 × 1.5 ≈ 44.145 J."
  },
  "answer": 44,
  "solving_time_estimate": 1,
  "required_concepts": [
    "Gravitational potential energy"
  ],
  "potential_errors": [
    "Using g = 10 instead of 9.81",
    "Forgetting to round the final answer"
  ]
}
""",

    2: """
{
  "analysis": "Design a moderate-level problem requiring exactly two reasoning steps and two intermediate results. The problem uses: (1) translational deceleration from friction a = μg, and (2) the stopping distance formula d = v0^2 / (2a). These two linked steps satisfy the Moderate complexity requirement.",
  "question": "A bowling ball slides on a horizontal lane with initial speed 7.0 m/s. The coefficient of sliding friction is μ = 0.25, and g = 9.81 m/s^2. Assuming friction is the only horizontal force, how far does the ball slide before stopping? Give your answer in meters, rounded to two decimal places.",
  "intermediate_results": {
    "step1_compute_deceleration": "Compute the friction-based deceleration a = μg = 0.25 × 9.81 ≈ 2.4525 m/s^2.",
    "step2_compute_distance": "Use d = v0^2 / (2a) = 7.0^2 / (2 × 2.4525) ≈ 9.99 m."
  },
  "answer": 9.99,
  "solving_time_estimate": 4,
  "required_concepts": [
    "Kinetic friction",
    "Constant-acceleration kinematics"
  ],
  "potential_errors": [
    "Using the wrong sign for deceleration",
    "Using mass in a = μg even though it cancels",
    "Using v^2 = v0^2 + 2ad with the wrong sign"
  ]
}
""",

    3: """
{
  "analysis": "Create a hard-level problem requiring a deep multi-step reasoning chain involving coupled translational and rotational dynamics. We require eight intermediate reasoning steps. A ball is slid without rotation; friction slows translation, increases rotation, and the ball enters pure rolling when v = ωr. Then compute the distance traveled until that moment. This problem uses many spans (a, α, torque, inertia, rolling constraint, motion equations) and satisfies the Hard definition with 8+ reasoning steps.",
  "question": "A solid bowling ball of mass 6 kg and radius 0.12 m is thrown along a horizontal lane with initial speed 12.0 m/s and no initial rotation. The coefficient of sliding friction is μ = 0.22, and g = 9.81 m/s^2. Friction slows translation and increases rotation until the ball reaches rolling without slipping. Treat the ball as a solid sphere with moment of inertia I = (2/5)mr^2. Assuming constant friction, how far does the ball travel before it first reaches pure rolling? Give your answer in meters, rounded to two decimal places.",
  "intermediate_results": {
    "step1_compute_linear_deceleration": "a = μg = 0.22 × 9.81 ≈ 2.1582 m/s^2.",
    "step2_compute_friction_force": "F = μmg = 0.22 × 6 × 9.81 ≈ 12.9492 N.",
    "step3_compute_torque": "τ = Fr = 12.9492 × 0.12 ≈ 1.5539 N·m.",
    "step4_compute_inertia": "I = (2/5)mr^2 = (2/5) × 6 × 0.12^2 ≈ 0.03456 kg·m^2.",
    "step5_compute_angular_acceleration": "α = τ / I ≈ 1.5539 / 0.03456 ≈ 44.9625 rad/s^2.",
    "step6_set_up_motion_equations": "v(t) = v0 - at, ω(t) = αt, with v0 = 12.0 m/s.",
    "step7_solve_for_time": "Rolling starts when v(t) = ω(t) r ⇒ 12.0 - 2.1582 t = 44.9625 t × 0.12. 12 = (2.1582 + 5.3955)t. Solve for t ≈ 1.5886 s.",
    "step8_compute_distance": "d = v0 t - 0.5 a t^2 = 12 × 1.5886 - 0.5 × 2.1582 × 1.5886^2 ≈ 19.0632 - 2.7231 ≈ 16.34 m."
  },
  "answer": 16.34,
  "solving_time_estimate": 15,
  "required_concepts": [
    "Kinetic friction",
    "Torque due to friction",
    "Moment of inertia of a solid sphere",
    "Coupled translational and rotational motion",
    "Rolling without slipping condition"
  ],
  "potential_errors": [
    "Using wrong sign conventions",
    "Forgetting that friction accelerates rotation while decelerating translation",
    "Using ω = v/r instead of v = ωr",
    "Dropping the factor (2/5) in I",
    "Incorrectly solving the equation for t",
    "Forgetting the 0.5 a t^2 term in distance"
  ]
}
"""
}


CHALLENGER_PROMPT_TEMPLATE = '''Your task is to generate a single self-contained question and its correct answer inspired by the given document.
The question must strictly satisfy both the difficulty level and the answer_type constraints.

You must output exactly one JSON object as specified below.  
All reasoning MUST be placed inside the "analysis" field of the JSON.

## Document
[BEGIN]
{text}
[END]

## Difficulty Level

You are given a target difficulty level:

- difficulty_id: {difficulty_id}

You must follow these operational definitions:

{definition}

Your generated question and solution process must match the target difficulty level as closely as possible.

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
4. The reasoning steps must match the difficulty level.
5. The question must combine the number of spans required by difficulty level.
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

# Example Output (do not copy; only follow structure; the content and answer type (Integer vs Float) in the example below are for structure demonstration only. The example is based on a generic Physics document. Your output must be based on the provided [Document] above and strictly follow the "difficulty_id" and "answer_type" assigned to you.)

{example}
'''


def _gen_single_item(text: str, idx: int, difficulty_id: int, answer_type: str, type: str):
    """构造一条样本（给定 text/doc 与目标难度/答案类型）。"""
    # Map numeric difficulty id to label used in the prompt
    id2level = {1: 'Easy', 2: 'Moderate', 3: 'Hard'}
    question = CHALLENGER_PROMPT_TEMPLATE.format(
        difficulty_id=id2level[difficulty_id],
        answer_type=answer_type,
        text=text,
        definition=id2def[difficulty_id],
        example=id2ICL[difficulty_id],
    )
    # print(question)
    return {
        "data_source": "questioner_given_difficulty_id",
        "prompt": [
            {
                "role": "user",
                "content": question,
            }
        ],
        "ability": type,
        "reward_model": {"style": "rule", "ground_truth": difficulty_id},
        "extra_info": {
            "split": "train",
            "index": idx,  # 原始文档 id，用于成组排序奖励
            "answer_type": answer_type,
            "text": text,
        },
    }


def process_batch_fn(batch, indices, difficulties=(1, 2, 3), k_per_doc=None, int_ratio=0.8):
    """
    将每条原始文档扩展为多个难度标注的样本，默认覆盖 1..3 全部难度。

    - difficulties: 可选难度集合，例如 (1,2,3)
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
        global_text_lenghs.append(len(batch["text"][i]))
        text = (batch["text"][i] or "")[:30000]
        doc_idx = indices[i] if isinstance(indices, list) else indices[i]

        # 选择该文档要生成的难度集合
        if k_per_doc is not None and k_per_doc < len(difficulties):
            diffs = random.sample(list(difficulties), k_per_doc)
        else:
            diffs = list(difficulties)

        for d in sorted(diffs):
            answer_type = "integer" if random.random() < int_ratio else "float"


            # NOTE : hard code判类别

            if batch['language_id_whole_page_fasttext'][i] is None:
                type = 'math'
            else:
                type = 'general'
            item = _gen_single_item(text=text, idx=doc_idx, difficulty_id=d, answer_type=answer_type, type=type)
            for k in out.keys():
                out[k].append(item[k])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path",
        default="/share_data/data1/fanshengda/DEvo/data/all_filter1212.parquet",
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/share_data/data1/fanshengda/DEvo/data/challenger_1212",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument(
        "--difficulties",
        default="1,2,3",
        help="Comma-separated difficulty ids to expand per document (default: all 1..3).",
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
    df = pd.read_parquet(args.local_dataset_path).head(20000)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=42)

    # 2. 按文档扩展多个难度版本，默认 1..3 全覆盖
    diffs = tuple(int(x) for x in re.split(r"[，,\s]+", args.difficulties.strip()) if x)
    train_dataset = dataset.map(
        lambda batch, idxs: process_batch_fn(batch, idxs, difficulties=diffs, k_per_doc=args.k_per_doc, int_ratio=args.int_ratio),
        with_indices=True,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print('da')
    # 3. 保存为 parquet
    os.makedirs(args.local_save_dir, exist_ok=True)
    save_path = os.path.join(args.local_save_dir, "train.parquet")
    train_dataset.to_parquet(save_path)
    print(f"Saved processed dataset to: {save_path}")
    print(f"Dataset size: {len(train_dataset)}")
