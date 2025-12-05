#!/bin/bash

# nohup bash scripts/difficulty_aware_questioner_train.sh /share_data/data1/models/Qwen/Qwen3-4B-Base /share_data/data1/models/Qwen/Qwen3-4B-Base qwen3-4b-difficulty_aware_questioner > /data3/workhome/fanshengda/DEvo/logs/difficulty_aware_questioner_train_new.log 2>&1 &
export STORAGE_PATH="/share_data/data1/fanshengda/DEvo/ckpts"
export HUGGINGFACENAME="AnIdealRing"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data3/workhome/fanshengda/DEvo:$PYTHONPATH
export INJECT_EXTRA_INFO_TO_GROUND_TRUTH=1
export VLLM_N_CANDIDATES=16


export ENABLE_DIFFICULTY_RANKING=0
export DEEPSEEK_MODEL="Qwen3-4B-Instruct-2507"
export DEEPSEEK_API_URL="http://10.0.1.10:8888/v1"
export DEEPSEEK_API_KEY="dada"

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
    data.max_prompt_length=10000 \
    data.max_response_length=8000 \
    data.train_files=/share_data/data1/fanshengda/DEvo/data/challenger_1204 \
    data.val_files=/share_data/data1/fanshengda/DEvo/data/challenger_1204 \
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
    worker.rollout.n=4 \
    worker.actor.global_batch_size=12 \
    data.rollout_batch_size=12 \
    worker.actor.ulysses_sequence_parallel_size=2 \
    trainer.max_steps=5000 \
    trainer.save_freq=50 \
    trainer.val_before_train=false


sleep 5

# 合并模型
echo "merging model"
python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/$save_path/global_step_5/actor

sleep 10

pkill python

echo "questioner training finished"
