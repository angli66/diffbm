import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import pad_img, NCC

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Hyperparameters
    max_disp = 128 - 1 # Inclusive
    block_size = 21
    focal_length = 920 # In pixel
    baseline_length = 0.0545 # In meter
    min_depth = 0.2
    max_depth = 2.0
    batch_size = 128 # Lower batch size if CUDA out of memory
    
    # Load images
    left = plt.imread("img/left.png") # Image should be of shape hxw, within range [0.0, 1.0]
    left_padded = pad_img(left, block_size)[..., None]
    left_padded = torch.Tensor(left_padded).permute(2, 0, 1).to(device)

    right = plt.imread("img/right.png") # Image should be of shape hxw, within range [0.0, 1.0]
    right_padded = pad_img(right, block_size)[..., None]
    right_padded = torch.Tensor(right_padded).permute(2, 0, 1).to(device)

    # Compute matching cost aray
    h, w = left.shape
    half = block_size // 2
    matching_cost = []
    for i in range(h // batch_size):
        kernels = []
        inputs = []
        for y in range(i * batch_size, (i + 1) * batch_size):
            for x in range(max_disp, w):
                kernel = left_padded[:, y:y+2*half+1, x:x+2*half+1]
                input = right_padded[:, y:y+2*half+1, x-max_disp:x+2*half+1]
                kernels.append(kernel)
                inputs.append(input)
        kernels = torch.stack(kernels)
        inputs = torch.stack(inputs)
        ncc = NCC(kernels)
        result = ncc(inputs).squeeze().flip(1)
        assert result.isnan().any() == False # If true, increase the trivial term in ncc.py
        matching_cost.append(result)
    
    # Height not divisible by batch size
    kernels = []
    inputs = []
    for y in range((h // batch_size) * batch_size, h):
        for x in range(max_disp, w):
            kernel = left_padded[:, y:y+2*half+1, x:x+2*half+1]
            input = right_padded[:, y:y+2*half+1, x-max_disp:x+2*half+1]
            kernels.append(kernel)
            inputs.append(input)
    kernels = torch.stack(kernels)
    inputs = torch.stack(inputs)
    ncc = NCC(kernels)
    result = ncc(inputs).squeeze().flip(1)
    assert result.isnan().any() == False # If true, increase the trivial term in ncc.py
    matching_cost.append(result)

    matching_cost = torch.cat(matching_cost, axis=0)

    # Sanity Check
    # disp = torch.argmax(matching_cost, dim=-1)
    # disp = disp.reshape(h, w - max_disp)
    # depth = focal_length * baseline_length / disp
    # torch.nan_to_num(depth, nan=0.0)
    # depth[depth < min_depth] = 0
    # depth[depth > max_depth] = 0
    # plt.imshow(depth.cpu(), cmap='jet')
    # plt.show()
