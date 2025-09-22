export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN=${HF_TOKEN}
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"


TASKS="sat_bench,mmsi_bench,mindcube,threedsr_bench_real,vsr"
TASK_SUFFIX="${TASKS//,/_}"
echo $TASK_SUFFIX

# accelerate launch --num_processes=2 --main_process_port 12399 -m lmms_eval \
#     --model=llava_onevision \
#     --model_args=pretrained=/mnt/world_foundational_model/gkz/ckpts/hub/models--lmms-lab--llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
#     --tasks=$TASKS \
#     --batch_size=1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \
#     --output_path ./baseline_wlm/

# accelerate launch --num_processes=2 --main_process_port 12399 -m lmms_eval \
#     --model qwen2_5_vl \
#     --model_args=pretrained=/mnt/world_foundational_model/gkz/ckpts/hub/Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=12845056,attn_implementation=flash_attention_2,interleave_visuals=False \
#     --tasks $TASKS \
#     --batch_size 1 \
#     --log_samples \
#     --log_samples_suffix llava_onevision \
#     --output_path ./baseline_wlm/

accelerate launch --num_processes 2 --main_process_port 12399 -m lmms_eval \
    --model internvl2_5 \
    --model_args=pretrained=/mnt/world_foundational_model/gkz/ckpts/hub/OpenGVLab/InternVL2_5-8B \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $TASK_SUFFIX \
    --output_path ./baseline_wlm/