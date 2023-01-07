import torch
from torch import nn, Tensor
from .image import ImageList

class AnchorGener(nn.Module):
    """
    anchor generator
    """
    def __init__(self, sizes=None, aspect_ratios=None):
        super(AnchorGener, self).__init__()

        if sizes is None:
            sizes = [128, 256, 512]
        if aspect_ratios is None:
            aspect_ratios = [0.5, 1.0, 2.0]

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.anchor_cache = {}
        self.num_anchors_per_location = len(sizes) * len(aspect_ratios)
        self.local_anchor = None

    def set_local_anchor(self, dtype: torch.dtype, device: torch.device):
        if self.local_anchor is not None and self.local_anchor.device == device:
            return self.local_anchor

        sizes = torch.as_tensor(self.sizes, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=dtype, device=device)
        height_ratio = torch.sqrt(aspect_ratios)
        width_ratio = 1.0 / height_ratio
        
        height = (height_ratio[:, None] * sizes[None, :]).view(-1)
        width = (width_ratio[:, None] * sizes[None, :]).view(-1)
        local_anchors = torch.stack([-width, -height, width, height], dim=1) / 2
        self.local_anchor = local_anchors.round()
        return self.local_anchor

    def get_global_anchor(self, grid_sizes, strides):
        key = str(grid_sizes) + str(strides)
        if key in self.anchor_cache:
            return self.anchor_cache[key]
        else:
            anchors = self.global_anchor(grid_sizes, strides)
            self.anchor_cache[key] = anchors
            return anchors

    def global_anchor(self, grid_sizes, strides):
        local_anchor = self.local_anchor
        assert local_anchor is not None

        grid_height, grid_width = grid_sizes
        stride_height, stride_width = strides
        device = local_anchor.device

        shift_height = torch.arange(0, grid_height, dtype=torch.float32, device=device) * stride_height
        shift_width = torch.arange(0, grid_width, dtype=torch.float32, device=device) * stride_width

        shift_height, shift_width = torch.meshgrid(shift_height, shift_width)
        shift_height = shift_height.reshape(-1)
        shift_width = shift_width.reshape(-1)
        shifts = torch.stack((shift_width, shift_height, shift_width, shift_height), dim=1)
        anchors = shifts.view(-1, 1, 4) + local_anchor.view(1, -1, 4)
        return anchors.reshape(-1, 4)

    def forward(self, image_list: ImageList, feature_map):
        image_size = image_list.tensors.shape[-2:]
        grid_size = feature_map.shape[-2:]
        dtype, device = image_list.tensors.dtype, image_list.tensors.device
        stride = [torch.tensor(image_size[0] // grid_size[0], dtype=torch.int64, device=device),
                  torch.tensor(image_size[1] // grid_size[1], dtype=torch.int64, device=device)]
        self.set_local_anchor(dtype, device)
        anchor_on_feature_map = self.get_global_anchor(grid_size, stride)
        anchors = []
        for i in range(len(image_list.image_sizes)):
            anchors.append(anchor_on_feature_map)
        self.anchor_cache.clear()
        return anchors
