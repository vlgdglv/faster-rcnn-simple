import torch
from torch import Tensor
from typing import List, Tuple


class ImageList(object):
    def __init__(self, tensors: Tensor, image_sizes, original_image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.original_image_sizes = original_image_sizes

    def to(self, device: torch.device) -> 'ImageList':
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes, self.original_image_sizes)
