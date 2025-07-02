import argparse
import os
import random
from warnings import simplefilter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import RMSO_ConvNeXt
from utils import fft_match_batch, Addnoise

# 除掉相关不必要警告
simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Configs')

    # 权重路径
    parser.add_argument('--resume',
                        default=
                        "weights/Pre-training-weight.pth",
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # Device options
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()

    if torch.cuda.is_available():
        cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    model = RMSO_ConvNeXt()
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print(f"=> no checkpoint found at {args.resume}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    model = model.to(device)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        data_ref = Image.open('image/opt3.png').convert('L')
        data_tem = Image.open('image/sar3.png').convert('L')
        Tensor_ref = transforms.ToTensor()(data_ref).squeeze()
        Tensor_tem = transforms.ToTensor()(data_tem).squeeze()

        Tensor_ref = Tensor_ref.unsqueeze(0).unsqueeze(0)
        Tensor_tem = Tensor_tem.unsqueeze(0).unsqueeze(0)
        out_ref, out_t = model(Tensor_ref.to(device=device), Tensor_tem.to(device=device))

        out = fft_match_batch(out_ref, out_t)

        for j, temp in enumerate(out.cpu()):
            # 获取最佳匹配点坐标
            index = np.unravel_index(temp.argmin(), temp.shape)
            y = index[0]
            x = index[1]

    fig, axes = plt.subplots(1, 4, figsize=(60, 30))  # 1行多列
    titles = ['Reference Image', 'Template Image', 'Noisy Image', 'Matching Image']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=60)
    axes[0].imshow(data_ref, cmap='gray')
    axes[1].imshow(data_tem, cmap='gray')
    axes[2].imshow(data_tNoise, cmap='gray')
    paste_img = data_ref.copy()
    paste_img.paste(data_tNoise, (x, y))
    axes[3].imshow(paste_img, cmap='gray')
    for ax in axes:
        ax.axis('off')
    plt.show()
    print("done!")
