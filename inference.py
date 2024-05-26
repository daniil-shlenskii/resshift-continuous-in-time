import cv2
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import yaml
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser

from utils import instantiate_from_config, prepare_model
from samplers import EulerSampler, HeunSampler


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        help="path to directory with low-resolution images"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="path to directory where high-resolution images will be stores"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to yaml file with config"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference"
    )
    parser.add_argument(
        "--lr_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="euler",
    )
    parser.add_argument(
        "--ro",
        type=int,
        default="1",
    )
    parser.add_argument(
        "--reverse_ro",
        action='store_true',
    )
    args = parser.parse_args()
    return args

class InferenceDataset(Dataset):
    def __init__(self, in_dir, lr_size):
        self.paths = list(Path(in_dir).iterdir())
        self.images = []
        for path in self.paths:
            im = cv2.imread(str(path))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            self.images.append(im)

        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3, inplace=False),
            T.Resize(lr_size, interpolation=T.InterpolationMode.BICUBIC),
        ])
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = str(self.paths[idx]).split("/")[-1]
        im = self.images[idx]
        im = self.transforms(im)
        return path, im

def get_timesteps(N, ro):
    timesteps = np.arange(N) / (N - 1)
    timesteps = (1 - timesteps)**ro
    return timesteps

def main():
    args = get_parser()

    with open(args.config_path) as file:
        config = yaml.safe_load(file)

    ae_config = config["autoencoder"]
    ae = instantiate_from_config(ae_config)
    ae = prepare_model(ae, ae_config).cuda()

    model_config = config["model"]
    model = instantiate_from_config(model_config)
    model = prepare_model(model, model_config).cuda()

    loader = DataLoader(
        InferenceDataset(args.in_dir, args.lr_size),
        batch_size=args.batch_size,
        num_workers=2,
    )

    if args.sampler == "euler":
        sampler = EulerSampler(ae=ae, x0_pred_fn=model, device="cuda")
    elif args.sampler == "heun":
        sampler = HeunSampler(ae=ae, x0_pred_fn=model, device="cuda")
    else:
        raise ValueError()
    
    subdir_name = f"{args.sampler}_{args.n_steps}_{args.ro}"
    if args.reverse_ro:
        subdir_name = subdir_name + "_reversed-ro"
    subdir_path = Path(args.out_dir) / subdir_name
    subdir_path.mkdir(exist_ok=True, parents=True)

    timesteps = list(get_timesteps(args.n_steps + 1, args.ro))
    if args.reverse_ro:
        timesteps = 1 - torch.tensor(timesteps[::-1])
    else:
        timesteps = torch.tensor(timesteps)
    
    for file_names, lq in loader:
        lq = lq.cuda()
        out = sampler(timesteps, lq).permute(0, 2, 3, 1).detach().cpu().numpy()
        out = out * 0.5 + 0.5
        for file_name, sr_image in zip(file_names, out):
            Image.fromarray(np.uint8(sr_image*255)).save(str(subdir_path / file_name))
        break

    print(f"Images saved into {subdir_path}")

if __name__ == "__main__":
    main()