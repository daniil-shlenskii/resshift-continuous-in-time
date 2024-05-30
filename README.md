
# ResShift continuous-in-time

In this repo we implement idea from the [ResShift](https://arxiv.org/abs/2307.12348) paper in the continuous setup, i.e. we change Markov Chain represented in the original paper with Stochstic Differential Equation. We investigate possibilities that such setup opens, namely, we utilize high order ODE solvers and consider different time discretizations, what results in better than in the paper. Details can be found in report (!link to the report!)

**Inference**
---
```bash
# Downloading weights
mkdir weights
wget https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth # autoencoder
wget gdown 1P17fgFhSSpL5mbhSND2KjDijxhZLavO_ # denoiser
```
After that you can immediately start working in the `playground/pgd.ipynb` file.

If you need to upscale a dataset of low-resolution images you need run the following line:

```bash
python inference.py --in_dir <low-res-dir> --out_dir <dir-to-put-upscales> --config_path <path-to_config> --batch_size 8 --ro 1
```
