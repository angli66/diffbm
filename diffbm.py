import matplotlib.pyplot as plt
import torch
from utils import block_matching

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    # Hyperparameters
    min_disp = 33 # max_depth = focal_length * baseline_length / min_disp
    max_disp = 128 # min_depth = focal_length * baseline_length / max_disp
    block_size = 21 # Block size for block matching
    focal_length = 920 # In pixel
    baseline_length = 0.0545 # In meter
    batch_size = 128 # Lower batch size if CUDA out of memory

    # Optimizable parameter
    temperature = torch.tensor([1000.], requires_grad=True).to(device) # Temperature for softargmax
    
    # Load images
    img_l = torch.Tensor(plt.imread("img/left.png")).to(device) # Image should be of shape hxw, within range [0.0, 1.0]
    img_r = torch.Tensor(plt.imread("img/right.png")).to(device) # Image should be of shape hxw, within range [0.0, 1.0]

    # Block matching
    disp = block_matching(
        img_l,
        img_r,
        min_disp=min_disp,
        max_disp=max_disp,
        block_size=block_size,
        temperature=temperature,
        batch_size=batch_size
    )

    # Disparity to depth conversion
    depth = focal_length * baseline_length / disp # Of shape [h, w-max_disp], viewed in left camera's frame

    # Sanity Check
    plt.imshow(depth.detach().cpu(), cmap='jet')
    plt.show()
