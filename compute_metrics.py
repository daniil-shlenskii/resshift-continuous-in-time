import cv2
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

import piq

import yaml
import json
from PIL import Image
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict

from utils import instantiate_from_config, prepare_model
from samplers import EulerSampler


def get_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="path to directory with super-resolved images"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        help="path to directory where high-resolution images will be stores"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="scores",
        help="path to directory where results will be stored"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference"
    )
    args = parser.parse_args()
    return args

class InferenceDataset(Dataset):
    def __init__(self, pred_dir, target_dir):
        file_names = [
            str(path).split("/")[-1]
            for path in Path(pred_dir).iterdir()
        ]

        self.preds, self.targets = [], []
        for file_name in file_names:
            pred = cv2.cvtColor(cv2.imread(f"{pred_dir}/{file_name}"), cv2.COLOR_BGR2RGB)
            target = cv2.cvtColor(cv2.imread(f"{target_dir}/{file_name}"), cv2.COLOR_BGR2RGB)
            self.preds.append(pred)
            self.targets.append(target)

        self.transforms = T.Compose([
            T.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.preds)

    def __getitem__(self, idx):
        pred = self.transforms(self.preds[idx])
        target = self.transforms(self.targets[idx])
        return pred, target

def main():
    args = get_parser()

    loader = DataLoader(
        InferenceDataset(args.pred_dir, args.target_dir),
        batch_size=args.batch_size,
        num_workers=2,
    )
    
    score_fns = {
        "lpips": piq.LPIPS(),
        "ssim": piq.SSIMLoss(data_range=1.),
        "psnr": piq.psnr,
    }
    scores = defaultdict(float)
    for pred, target in loader:
        for key, score_fn in score_fns.items():
            scores[key] += score_fn(pred, target).item() * len(pred) / len(loader.dataset)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    result_path = result_dir / args.target_dir.split("/")[-1]
    
    with open(result_path, 'w') as file:
        json.dump(scores, file)

    print(f"Scores save to {result_path}")

if __name__ == "__main__":
    main()