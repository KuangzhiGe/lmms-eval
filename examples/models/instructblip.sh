export WANDB_API_KEY="cfbb7b5b972619513ee861d88956b8e497dc71da"
export http_proxy=http://192.168.32.28:18000
export https_proxy=http://192.168.32.28:18000
export HF_TOKEN=${HF_TOKEN}
export HF_HOME="/mnt/world_foundational_model/gkz/ckpts"

cd /mnt/world_foundational_model/gkz/lmms-eval
python3 -m pip install -e .;

python3 -m pip install transformers --upgrade;

CKPT_PATH="/mnt/world_foundational_model/gkz/lmms-eval/baselines/Salesforce/instructblip-vicuna-7b"
TASK="gqa"
echo $TASK
TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX

accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model instructblip \
    --model_args pretrained=$CKPT_PATH \
    --tasks $TASK \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix instructblip \
    --output_path ./logs/