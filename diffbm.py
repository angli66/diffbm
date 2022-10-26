import numpy as np
import matplotlib.pyplot as plt
import torch
from ncc import NCC

def pad_img(img, block_size):
    half = block_size // 2
    padding = np.zeros((img.shape[0], half))
    img = np.concatenate([padding, img, padding], axis=1)
    padding = np.zeros((half, img.shape[1]))
    img = np.concatenate([padding, img, padding], axis=0)

    return img


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # # Hyperparameters
    max_disp = 96 - 1 #inclusive
    block_size = 21
    # min_depth = 0.2
    # max_depth = 2.0
    # f = 0
    # b = 0

    # Other parameters
    half = block_size // 2

    left = plt.imread("img/left.png")
    h, w = left.shape
    left_padded = pad_img(left, block_size)[..., None]
    left_padded = torch.Tensor(left_padded).permute(2, 0, 1).to(device)

    # output = []
    # for y in range(h):
    #     output_row = []
    #     for x in range(max_disp, w):
    #         left_input = left_padded[:, y:y+2*half+1, x:x+2*half+1]
    #         right_input = left_padded[:, y:y+2*half+1, x-max_disp:x+2*half+1]

    #         ncc = NCC(left_input)
    #         match_result = ncc(right_input[None, ...]).squeeze()
    #         match_result = torch.flip(match_result, [0])
    #         output_row.append(match_result)

    #         print(match_result.shape)
    #         print(f"Best match: ({torch.argmax(match_result).item()})")
    #         print("Min ncc:", torch.min(match_result).item())
    #         print("Max ncc:", torch.max(match_result).item())
    #         print("Contains nan:", torch.isnan(match_result).any().item())

    #     output_row = torch.stack(output_row)
    #     output.append(output_row)

    # output = torch.stack(output)
    # print(output.shape)

    kernels = []
    inputs = []
    for y in range(1): # range(h)
        for x in range(max_disp, w):
            kernel = left_padded[:, y:y+2*half+1, x:x+2*half+1]
            input = left_padded[:, y:y+2*half+1, x-max_disp:x+2*half+1]
            kernels.append(kernel)
            inputs.append(input)
    kernels = torch.stack(kernels)
    inputs = torch.stack(inputs)

    print(kernels.shape)
    print(inputs.shape)
