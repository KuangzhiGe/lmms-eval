export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN=${HF_TOKEN}
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git

accelerate launch --num_processes 2 --main_process_port 12345 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks physicalmind \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_7B \
    --output_path ./logs/