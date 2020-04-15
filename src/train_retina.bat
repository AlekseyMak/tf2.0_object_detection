MODEL_DIR="..\models"
python ..\models\official\vision\detection\main.py ^
  --strategy_type=one_device ^
  --num_gpus=1 ^
  --model_dir="${MODEL_DIR?}" ^
  --mode=train ^
  --config_file="my_retinanet.yaml"