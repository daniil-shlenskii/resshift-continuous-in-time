LOWRES_DIR="testdata/Val_SR/lq"
TARGET_DIR="testdata/Val_SR/gt"
UPSCALES_DIR="upscales/Val_SR"
CONFIG_PATH="configs/inference.yaml"
RESULT_DIR="results"

for ro in 2 3
do
    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size 32 --ro $ro
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/euler_15_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size 32

    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size 32 --ro $ro --reverse_ro
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/euler_15_${ro}_reversed-ro --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size 32

    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size 32 --ro $ro --sampler heun --n_steps 8
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/heun_8_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size 32

    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size 32 --ro $ro --sampler heun --n_steps 8 --reverse_ro
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/heun_8_${ro}_reversed-ro --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size 32
done