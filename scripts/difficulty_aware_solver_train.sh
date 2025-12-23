# nohup bash scripts/difficulty_aware_solver_train.sh /share_data/data1/models/Qwen/Qwen3-4B-Base /share_data/data1/fanshengda/DEvo/data/solver_1221/solver_questioner_350_train.parquet qwen3-4b-difficulty_aware_solver_1223 > /data3/workhome/fanshengda/DEvo/logs/difficulty_aware_solver_train_1223.log 2>&1 &
export STORAGE_PATH="/share_data/data1/fanshengda/DEvo/ckpts"
export HUGGINGFACENAME="AnIdealRing"
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=/data3/workhome/fanshengda/DEvo:$PYTHONPATH
export INJECT_EXTRA_INFO_TO_GROUND_TRUTH=0

solver_model_path=$1
solver_train_file=$2
experiment_name=$3

echo $STORAGE_PATH

echo "start train solver $experiment_name $solver_model_path $questioner_model_path" 

export VLLM_DISABLE_COMPILE_CACHE=1



python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.max_response_length=4096 \
    data.shuffle=false \
    data.rollout_batch_size=128 \
    worker.actor.global_batch_size=128 \
    data.train_files=${solver_train_file} \
    worker.actor.model.model_path=$solver_model_path \
    trainer.experiment_name=${experiment_name} \
    trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
    trainer.self_vote_ratio_use_all=false \
    trainer.dump_rollout_n=32 \
    trainer.dump_rollout_every=1 \
    trainer.dump_rollout_path=${STORAGE_PATH}/models/${experiment_name}/rollouts.jsonl \
    trainer.total_epochs=1 \
    trainer.max_steps=512 \
    trainer.save_limit=10 \
    data.format_prompt=./examples/format_prompt/solver.jinja \
    data.train_prompt_key=prompt\
    data.train_answer_key=reward_model \
    data.val_prompt_key=problem \
    data.val_answer_key=answer \
    trainer.val_freq=4 \
    trainer.save_freq=64 \
    algorithm.kl_coef=0 \
    worker.rollout.n=8 \
    algorithm.norm_adv_by_std_in_grpo=false \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.reward.reward_function=./examples/reward_function/difficulty_aware_solver.py:compute_score \
    worker.reward.reward_function_kwargs.solver_label_mode=self_vote \
    worker.reward.reward_function_kwargs.label_prompt_key=text_prompt \
    worker.reward.reward_function_kwargs.label_vote_threshold=0.3 \
    worker.reward.reward_function_kwargs.label_n=10 \
    worker.reward.reward_function_kwargs.label_temperature=1.0 \
    worker.reward.reward_function_kwargs.label_top_p=0.95


#echo ""merging model
#python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_55/actor

# python scripts/model_merger.py --local_dir /share_data/data1/fanshengda/DEvo/ckpts/models/qwen3-4b-difficulty_aware_solver_1221/global_step_192/actor/
#
#sleep 10
#
#echo "solver training finished"
#
#bash evaluation/evaluate.bash ${STORAGE_PATH}/models/${experiment_name}/global_step_15/actor/huggingface
#bash evaluation/evaluate.bash /share_data/data1/fanshengda/DEvo/ckpts/models/qwen3-4b-difficulty_aware_solver_1221/global_step_192/actor/huggingface





# python evaluation/difficulty_aware_results_recheck.py --result_path /share_data/data1/fanshengda/DEvo/ckpts/evaluation/_share_data_data1_fanshengda_DEvo_ckpts_models_qwen3-4b-difficulty_aware_solver_1221_global_step_192_actor_huggingface
