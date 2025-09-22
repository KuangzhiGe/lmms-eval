export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN=${HF_TOKEN}
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git --index-url https://pypi.org/simple
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git --index-url https://pypi.org/simple

accelerate launch --num_processes=8 --main_process_port 12399 -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=sat_bench,mmsi_bench,mindcube,threedsr_bench_real,vsr \
    --batch_size=1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs/