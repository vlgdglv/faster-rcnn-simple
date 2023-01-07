import torch
import math
from torch import nn, Tensor
from typing import List

def recover_boxes(boxes, from_size, to_size):
    ratios = [
        torch.tensor(t_size, dtype=torch.float32, device=boxes.device) /
        torch.tensor(f_size, dtype=torch.float32, device=boxes.device)
        for t_size, f_size in zip(to_size, from_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


def batch_up(images) -> Tensor:
    # shape: List[int], be like [3, 800, 1333]
    shape_list = [list(img.shape) for img in images]
    maxes = shape_list[0]
    for rest in shape_list[1:]:
        for idx, x in enumerate(rest):
            maxes[idx] = max(maxes[idx], x)
    shape = maxes

    # avoid overflow
    stride = 32.0
    shape[1] = int(math.ceil(float(shape[1]) / stride) * stride)
    shape[2] = int(math.ceil(float(shape[2]) / stride) * stride)

    batch_shape = [len(images)] + shape
    batch_images = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batch_images):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batch_images


def box_area(boxes):
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou


def remove_small_boxes(boxes, min_size):
    width, height = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    result = (width >= min_size) & (height >= min_size)
    return torch.where(result)[0]


def clip_box_within_image(boxes, image_size):
    dim = boxes.dim()
    height, width = image_size[0], image_size[1]
    boxes_x = boxes[..., 0::2].clamp(min=0, max=width)
    boxes_y = boxes[..., 1::2].clamp(min=0, max=height)
    cliped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
    return cliped_boxes.reshape(boxes.shape)


def encode_box(ref, pro, weights=None):
    if weights is None:
        wx = 1.0
        wy = 1.0
        ww = 1.0
        wh = 1.0
    else:
        wx, wy, ww, wh = weights
    
    pro_xmin = pro[:, 0].unsqueeze(1)
    pro_xmax = pro[:, 2].unsqueeze(1)
    pro_ymin = pro[:, 1].unsqueeze(1)
    pro_ymax = pro[:, 3].unsqueeze(1)

    ref_xmin = ref[:, 0].unsqueeze(1)
    ref_xmax = ref[:, 2].unsqueeze(1)
    ref_ymin = ref[:, 1].unsqueeze(1)
    ref_ymax = ref[:, 3].unsqueeze(1)

    pro_width = pro_xmax - pro_xmin
    pro_height = pro_ymax - pro_ymin
    pro_center_x = pro_xmin + pro_width * 0.5
    pro_center_y = pro_ymin + pro_height * 0.5

    ref_width = ref_xmax - ref_xmin
    ref_height = ref_ymax - ref_ymin
    ref_center_x = ref_xmin + ref_width * 0.5
    ref_center_y = ref_ymin + ref_height * 0.5

    dx = wx * (ref_center_x - pro_center_x) / pro_width
    dy = wy * (ref_center_y - pro_center_y) / pro_height
    dw = ww * torch.log(ref_width / pro_width)
    dh = wh * torch.log(ref_height / pro_height)

    reg_targets = torch.cat((dx, dy, dw, dh), dim=1)
    return reg_targets


def decode_bbox(bbox_deltas, bboxes, weights=None):
    if weights is None:
        wx = 1.0
        wy = 1.0
        ww = 1.0
        wh = 1.0
    else:
        wx, wy, ww, wh = weights
        
    boxes_per_image = [b.size(0) for b in bboxes]
    bboxes = torch.cat(bboxes, dim=0).to(bbox_deltas.device)
    total_boxes = 0
    for num in boxes_per_image:
        total_boxes += num
    if total_boxes > 0:
        bbox_deltas = bbox_deltas.reshape(total_boxes, -1)
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    center_x = bboxes[:, 0] + widths * 0.5
    center_y = bboxes[:, 1] + heights * 0.5

    dx = bbox_deltas[:, 0::4] / wx
    dy = bbox_deltas[:, 1::4] / wy
    dw = bbox_deltas[:, 2::4] / ww
    dh = bbox_deltas[:, 3::4] / wh
    clamp_max = math.log(1000. / 16)
    dw = torch.clamp(dw, max=clamp_max)
    dh = torch.clamp(dh, max=clamp_max)

    pred_ctr_x = dx * widths[:, None] + center_x[:, None]
    pred_ctr_y = dy * heights[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    _5 = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device)
    pred_xmin = pred_ctr_x - pred_w * _5
    pred_xmax = pred_ctr_x + pred_w * _5
    pred_ymin = pred_ctr_y - pred_h * _5
    pred_ymax = pred_ctr_y + pred_h * _5

    pred_bboxes = torch.stack((pred_xmin, pred_ymin, pred_xmax, pred_ymax), dim=2).flatten(1)
    if total_boxes > 0:
        pred_bboxes = pred_bboxes.reshape(total_boxes, -1, 4)
    return pred_bboxes


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def smooth_l1_loss(x, target, beta=1./9):
    n = torch.abs(x - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    return loss.sum()


def check_images(images):
    for img in images:
        assert len(img.shape[-2:]) == 2


def check_targets(targets):
    assert targets is not None
    for idx, target in enumerate(targets):
        boxes = target['boxes']
        assert isinstance(boxes, torch.Tensor)
        assert len(boxes.shape) == 2 and boxes.shape[-1] == 4
        degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
        if degenerate_boxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
            degen_bb = boxes[bb_idx].tolist()
            print("[FATAL] All bounding boxes should have positive height and width.\n"
                  " Found invalid box {} for target at index {}.".format(degen_bb, idx))
            return False
    return True
