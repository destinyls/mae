from audioop import avg
import torch
import torch.nn as nn

from torch._six import container_abcs
from itertools import repeat

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class DynamicMasking(nn.Module):
    def __init__(self, img_size, patch_size, mask_ratio=0.75):
        super(DynamicMasking, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.len_keep = int(num_patches * (1 - mask_ratio))

        self.avg_pool = nn.Conv2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, 196, 3]
        avg_x = self.avg_pool(x).flatten(2).transpose(1, 2)
        # [B, 196]
        avg_x = torch.mean(avg_x, dim=-1)
        patch_max = torch.max(avg_x, dim=1)[0]
        patch_min = torch.min(avg_x, dim=1)[0]
        # shape: [B, 196] | range: [0, 1]
        avg_x = (avg_x - patch_min) / (patch_max - patch_min)

        



        ids_shuffle = torch.argsort(avg_x, dim=1)

 

