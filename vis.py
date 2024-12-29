import warnings

warnings.filterwarnings('ignore')
import argparse
import os
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
import wandb
from scipy.optimize import linear_sum_assignment
from configs.basic_cfg import get_cfg
from models.MergeMRDD import MMRDD
from sklearn import metrics
from utils.datatool import (get_val_transformations,
                            add_sp_noise,
                            get_mask_train_dataset,
                            get_train_dataset)



@torch.no_grad()
def generate(val_dataloader, model, device, noise_prob=None):
    outputs = []
    for Xs, target in val_dataloader:
        if noise_prob:
            Xs = [add_sp_noise(x, noise_prob).to(device) for x in Xs]
        else:
            Xs = [x.to(device) for x in Xs]

        plt.figure(figsize=(3 * 2, 4 * 2))

        ncols = 2
        nrows = 5  # 根据 batch_size 计算行数
        for i in range(2 * 5):
            v = i % 2
            idx = i // 2
            image_tensor = Xs[v][idx].cpu()
            image_tensor = image_tensor.permute(1, 2, 0)
            image_tensor = image_tensor.clamp(0, 1)
            image_array = (image_tensor.numpy() * 255).astype(np.uint8)
            # 计算当前图像所在的子图位置
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(image_array)
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        out = model.generate(Xs)  # Tensor, list, list

        return out

    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args



def main():
    # Load arguments.
    args = parse_args()
    config = get_cfg(args.config_file)
    device = torch.device(f'cuda:{config.train.devices[0]}')

    val_transformations = get_val_transformations(config)
    train_set = get_train_dataset(config, val_transformations)
    train_dataloader = DataLoader(train_set,
                                num_workers=config.train.num_workers,
                                batch_size=config.train.batch_size,
                                sampler=None,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    mask_train_set = get_mask_train_dataset(config, val_transformations)
    mask_train_dataloader = DataLoader(mask_train_set,
                                num_workers=config.train.num_workers,
                                batch_size=config.train.batch_size,
                                sampler=None,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)

    run_times = 10
    n_clusters = config.dataset.class_num
    need_classification = False

    model_path = config.eval.model_path
    model = MMRDD(
        config=config,
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # summary(model)

    model = model.to(device)
    print(f'Use: {device}')

    model.eval()

    out = generate(train_dataloader, model, device, config.eval.noise_prob)

    ncols = 2
    nrows = config.train.batch_size  # 根据 batch_size 计算行数
    plt.figure(figsize=(ncols * 2, nrows * 2))
    for i in range(2 * 5):
        v = i % 2
        idx = i // 2
        image_tensor = out[v][idx].cpu()
        image_tensor = image_tensor.permute(1, 2, 0)
        image_tensor = image_tensor.clamp(0, 1)
        image_array = (image_tensor.numpy() * 255).astype(np.uint8)
        # 计算当前图像所在的子图位置
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(image_array)
        plt.axis('off')

    plt.tight_layout()
    plt.show()



    exit()


if __name__ == '__main__':
    main()