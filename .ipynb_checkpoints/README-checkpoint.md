
# ResShift continuous-in-time

In this repo we implement idea from the [ResShift](https://arxiv.org/abs/2307.12348) paper in the continuous setup, i.e. we change Markov Chain represented in the original paper with Stochstic Differential Equation. We investigate possibilities that such setup opens, namely, we utilize high order ODE solvers and consider different time discretizations, what results in better than in the paper. Details can be found in report (!link to the report!)

**Inference**
---
```bash
# Downloading weights
mkdir weights
cd weights
wget https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth # autoencoder
gdown 1P17fgFhSSpL5mbhSND2KjDijxhZLavO_ # denoiser
cd ..
```
After that you can immediately start working in the `playground/pgd.ipynb` file.

If you need to upscale a dataset of low-resolution images you need run the following line:

```bash
python inference.py --in_dir <low-res-dir> --out_dir <dir-to-put-upscales> --config_path <path-to_config> --batch_size 8 --ro 1
```

**Training**
---
In case you want to train the model on your data put your images in train_data directory and run the following commands:

```bash
mkdir weights
cd weights
wget https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth # autoencoder
wget https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth # denoiser
cd ..
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=0 python main.py --cfg_path configs/realsr_swinunet_realesrgan256.yaml --save_dir results_train
```
