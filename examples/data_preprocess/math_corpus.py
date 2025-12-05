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
    1: """```json
{
  "analysis": "I will design a very simple one-step numerical question based on the linear deceleration of the bowling ball due to kinetic friction. The document states that a bowling ball with coefficient of sliding friction μ = 0.52 and gravitational acceleration g = 9.81 m/s^2 experiences a friction force μmg in the horizontal direction. From Newton's second law, μmg = ma, so the magnitude of the linear deceleration is a = μg. This uses a single key span (the definition of μ and g) and one algebraic step, matching the Easy level: 1–2 spans and 1–2 reasoning steps. The answer is a float with two decimal places, which satisfies the numeric answer_type requirement.",
  "question": "A bowling ball is slid along a horizontal lane with a coefficient of sliding friction of 0.52. Take the acceleration due to gravity as 9.81 m/s^2. Assuming friction is the only horizontal force, what is the magnitude of the ball's initial linear deceleration in m/s^2? Give your answer correct to two decimal places.",
  "intermediate_results": {
    "step1_use_friction_law": "The kinetic friction force is F_f = μmg, where μ = 0.52 and g = 9.81 m/s^2.",
    "step2_apply_newton_second_law": "Since F_f = ma and friction acts opposite the motion, the magnitude of the deceleration is a = F_f/m = μg = 0.52 × 9.81.",
    "step3_compute_numeric_value": "Compute a ≈ 0.52 × 9.81 ≈ 5.10 m/s^2."
  },
  "answer": 5.10,
  "solving_time_estimate": 3,
  "required_concepts": [
    "Kinetic friction",
    "Coefficient of friction",
    "Newton's second law",
    "Basic multiplication"
  ],
  "potential_errors": [
    "Using g = 10 instead of 9.81 without being instructed to approximate",
    "Forgetting that the answer is a deceleration and should be positive in magnitude",
    "Multiplying 0.52 and 9.81 incorrectly"
  ]
}
```""",
    2: """```json
{
  "analysis": "I will design a moderate-difficulty question that asks for the initial angular acceleration of the bowling ball. The document provides: the diameter of the ball (28 cm, so radius r = 0.14 m), the coefficient of sliding friction μ = 0.52, gravitational acceleration g = 9.81 m/s^2, and the moment of inertia for a solid sphere I_cm = (2/5)mr^2. To find the angular acceleration, I combine at least two distinct spans: the friction force μmg, the torque τ = F_f r, and the relation τ = Iα. Substituting I = (2/5)mr^2, I obtain α = (5μg)/(2r). This chain involves about 2–3 reasoning steps and uses multiple pieces of information from the document, matching the Moderate level.",
  "question": "A solid bowling ball has a diameter of 28 cm and slides along a horizontal lane without initial rotation. The coefficient of sliding friction between the ball and the lane is 0.52, the acceleration due to gravity is 9.81 m/s^2, and the ball can be modeled as a solid sphere with moment of inertia I = (2/5)mr^2 about its center. Using these values, what is the magnitude of the ball's initial angular acceleration in rad/s^2? Round your answer to the nearest whole number.",
  "intermediate_results": {
    \"step1_extract_radius_and_inertia\": \"From the diameter 28 cm, the radius is r = 0.14 m. For a solid sphere, I = (2/5)mr^2.\",
    \"step2_compute_friction_torque\": \"The friction force is F_f = μmg with μ = 0.52, so the torque about the center is τ = F_f r = μmgr.\",
    \"step3_apply_rotational_dynamics\": \"Using τ = Iα and I = (2/5)mr^2, we get μmgr = (2/5)mr^2 α, so α = (5μg)/(2r).\",
    \"step4_numeric_evaluation\": \"Compute α ≈ (5 × 0.52 × 9.81) / (2 × 0.14) ≈ 91 rad/s^2.\"
  },
  "answer": 91,
  "solving_time_estimate": 6,
  "required_concepts": [
    "Torque from friction",
    "Moment of inertia of a solid sphere",
    "Rotational form of Newton's second law",
    "Algebraic manipulation"
  ],
  "potential_errors": [
    "Using the diameter instead of the radius in the inertia formula",
    "Forgetting the factor (2/5) in I = (2/5)mr^2",
    "Dropping the radius r when computing the torque",
    "Rounding too early and getting a noticeably different integer"
  ]
}
```
""",
    3: """```json
{
  "analysis": "I will create a hard question that requires several linked steps and uses multiple distant spans: the expression for linear deceleration a = μg, the rotational equation for angular acceleration α = (5μg)/(2r), and the rolling-without-slipping condition v = ωr. The bowling ball of radius r = 0.14 m has initial linear speed v_0 = 11 m/s and no initial rotation. While it is sliding, the linear speed decreases with constant deceleration a, and the angular speed increases with constant angular acceleration α. We have v(t) = v_0 - at and ω(t) = αt. The ball begins to roll without slipping when v(t) = ω(t) r, so v_0 - at = αtr. Solving for t gives t = v_0 / (a + αr). Substituting a = μg and α = (5μg)/(2r) leads to a multi-step computation. This uses the friction law, rotational dynamics, and the rolling condition together, requires about 5 or more reasoning steps, and therefore matches the Hard difficulty level.",
  "question": "A solid bowling ball with diameter 28 cm is slid along a horizontal lane with initial speed 11 m/s and no initial rotation. The coefficient of sliding friction between the ball and the lane is 0.52, the acceleration due to gravity is 9.81 m/s^2, and the moment of inertia of the ball about its center is I = (2/5)mr^2, where r is the radius of the ball. While it is sliding, friction both slows the translational motion and spins the ball up. Assuming the friction force and accelerations remain constant, how long after it is released does the ball first begin to roll without slipping? Give your answer in seconds, correct to two decimal places.",
  "intermediate_results": {
    "step1_compute_radius_and_deceleration": "The radius is r = 0.28 / 2 = 0.14 m. The linear deceleration from friction is a = μg = 0.52 × 9.81.",
    "step2_compute_angular_acceleration": "The friction force is F_f = μmg, giving a torque τ = F_f r. Using I = (2/5)mr^2 and τ = Iα, we get α = (5μg)/(2r).",
    "step3_write_kinematic_equations": "With initial speed v_0 = 11 m/s and no initial rotation, the linear speed is v(t) = v_0 - at and the angular speed is ω(t) = αt.",
    "step4_apply_rolling_condition": "Rolling without slipping begins when v(t) = ω(t) r, so v_0 - at = αtr. Rearranging gives v_0 = t(a + αr), hence t = v_0 / (a + αr).",
    "step5_numeric_substitution": "Compute a = 0.52 × 9.81 ≈ 5.10 m/s^2 and α ≈ (5 × 0.52 × 9.81)/(2 × 0.14) ≈ 91.1 rad/s^2. Then a + αr ≈ 5.10 + 91.1 × 0.14 and t ≈ 11 / (5.10 + 91.1 × 0.14) ≈ 0.62 s.",
    "step6_check_units_and_reasonableness": "The time is positive, on the order of a fraction of a second, and the condition v = ωr is satisfied at this instant, so the result is physically reasonable."
  },
  "answer": 0.62,
  "solving_time_estimate": 10,
  "required_concepts": [
    "Translational and rotational motion under friction",
    "Rolling without slipping condition v = ωr",
    "Moment of inertia of a solid sphere",
    "Coupled linear and angular kinematics",
    "Algebraic equation solving"
  ],
  "potential_errors": [
    "Forgetting to convert the diameter to radius before using r in formulas",
    "Using v = ω instead of v = ωr as the rolling condition",
    "Using a = μmg directly instead of dividing by mass to get the acceleration",
    "Dropping the factor (2/5) in the moment of inertia",
    "Making an algebra mistake when solving v_0 - at = αtr for t",
    "Rounding intermediate values too aggressively, leading to a noticeably different final time"
  ]
}
```
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

# Example Output (do not copy; only follow structure)

{example}
'''


def _gen_single_item(text: str, idx: int, difficulty_id: int, answer_type: str):
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
        "ability": "math",
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
            item = _gen_single_item(text=text, idx=doc_idx, difficulty_id=d, answer_type=answer_type)
            for k in out.keys():
                out[k].append(item[k])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_dataset_path",
        default="/share_data/data1/fanshengda/DEvo/data/filter1202.parquet",
        help="The local path to the raw dataset, if it exists.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/share_data/data1/fanshengda/DEvo/data/challenger_1204",
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
    df = pd.read_parquet(args.local_dataset_path).head(10000)
    dataset = Dataset.from_pandas(df)

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
