export XFL_CONFIG=experiments/config/insertanything.yaml

echo $XFL_CONFIG
export TOKENIZERS_PARALLELISM=true
# CUDA_VISIBLE_DEVICES=0
accelerate launch --main_process_port 41353  -m src.train.train