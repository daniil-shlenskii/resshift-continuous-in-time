OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=3

POSTFIX=_model_50k_9
CONFIG_POSTFIX=_model_50k

LOWRES_DIR=imagenet_256_test_lowres
TARGET_DIR=imagenet_256_test
UPSCALES_DIR=upscales${POSTFIX}
CONFIG_PATH=configs/inference${CONFIG_POSTFIX}.yaml
RESULT_DIR=results${POSTFIX}

NFE=9
BSIZE=96


ro=1
python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --n_steps $NFE --n_steps $NFE
python compute_metrics.py --pred_dir ${UPSCALES_DIR}/euler_${NFE}_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE 

for ro in 2 4 8
do
    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --n_steps $NFE
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/euler_${NFE}_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE

    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --reverse_ro --n_steps $NFE
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/euler_${NFE}_${ro}_reversed-ro --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE
done


NFE=5

ro=1
python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --n_steps $NFE --n_steps $NFE --sampler heun
python compute_metrics.py --pred_dir ${UPSCALES_DIR}/heun_${NFE}_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE 

for ro in 2 4 8
do
    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --n_steps $NFE  --sampler heun
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/heun_${NFE}_${ro} --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE

    python inference.py --in_dir $LOWRES_DIR --out_dir $UPSCALES_DIR --config_path $CONFIG_PATH --batch_size $BSIZE --ro $ro --reverse_ro --n_steps $NFE --sampler heun
    python compute_metrics.py --pred_dir ${UPSCALES_DIR}/heun_${NFE}_${ro}_reversed-ro --target_dir $TARGET_DIR --result_dir $RESULT_DIR --batch_size $BSIZE
done