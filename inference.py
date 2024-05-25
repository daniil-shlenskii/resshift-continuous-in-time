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
from samplers import EulerSampler


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

    sampler = EulerSampler(ae=ae, x0_pred_fn=model, device="cuda")
    timesteps = torch.tensor(list(np.linspace(0, 1, args.n_steps + 1))[::-1])
    
    for file_names, lq in loader:
        lq = lq.cuda()
        out = sampler(timesteps, lq).permute(0, 2, 3, 1).detach().cpu().numpy()
        out = out * 0.5 + 0.5
        for file_name, sr_image in zip(file_names, out):
            Image.fromarray(np.uint8(sr_image*255)).save(str(Path("logs") / file_name))

if __name__ == "__main__":
    main()