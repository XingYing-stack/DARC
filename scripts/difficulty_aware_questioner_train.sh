#!/bin/bash

# nohup bash scripts/difficulty_aware_questioner_train.sh /share_data/data1/models/Qwen/Qwen3-4B-Base /share_data/data1/models/Qwen/Qwen3-4B-Base qwen3-4b-difficulty_aware_questioner_1219 > /data3/workhome/fanshengda/DEvo/logs/difficulty_aware_questioner_train_1219.log 2>&1 &
export STORAGE_PATH="/share_data/data1/fanshengda/DEvo/ckpts"
export HUGGINGFACENAME="AnIdealRing"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data3/workhome/fanshengda/DEvo:$PYTHONPATH
export INJECT_EXTRA_INFO_TO_GROUND_TRUTH=1
export VLLM_N_CANDIDATES=10
export USE_TEXT_SOLVER_FOR_ANSWER=1
export ENABLE_COPY_PENALTY=1
export ENABLE_DIFFICULTY_RANKING=1
export DEEPSEEK_MODEL="gpt-4o-mini"
export DEEPSEEK_API_URL="https://toollearning.cn/v1"
export DEEPSEEK_API_KEY="sk-abHpUvVt7LLnmEyxCe17021d8e774e97Bd7aA9Bc4f2b1076"

solver_model_path=$1
questioner_model_path=$2
save_path=$3
echo "save_path: $save_path"
# 生成唯一 RUN_ID
RUN_ID=$(date +%s%N)
export RUN_ID

echo "RUN_ID=$RUN_ID"

# 启动 vllm 服务（记录 PID）
bash vllm_service_init/start_difficulty_aware.sh $solver_model_path $RUN_ID
echo "vLLM services started with RUN_ID=$RUN_ID"

# 开始训练 Questioner
echo "Start training difficulty_aware questioner: $questioner_model_path -> $save_path"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_prompt_length=13000 \
    data.max_response_length=3000 \
    data.train_files=/share_data/data1/fanshengda/DEvo/data/challenger_1219 \
    data.val_files=/share_data/data1/fanshengda/DEvo/data/challenger_1219 \
    data.shuffle=false \
    data.prompt_key=prompt \
    data.answer_key=reward_model \
    data.format_prompt=null \
    worker.rollout.max_num_batched_tokens=24000 \
    worker.actor.model.model_path=$questioner_model_path \
    trainer.experiment_name=$save_path \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
    trainer.total_epochs=1 \
    worker.reward.reward_function=./examples/reward_function/difficulty_aware_questioner.py:compute_score \
    trainer.val_freq=-1 \
    trainer.n_gpus_per_node=4 \
    worker.rollout.n=8 \
    worker.actor.global_batch_size=16 \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    data.rollout_batch_size=16 \
    worker.actor.ulysses_sequence_parallel_size=4 \
    trainer.max_steps=5000 \
    trainer.save_freq=50 \
    trainer.val_before_train=false


#sleep 5
#
## 合并模型
#echo "merging model"
#python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor
#
#sleep 10
#
#pkill python
#
#echo "questioner training finished"
