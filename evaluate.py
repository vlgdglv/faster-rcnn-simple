import numpy as np
import torch

def eval_voc(results, targets, iou_thresh, num_classes):
    for cls in range(1, num_classes+1):
        cls_results = []
        cls_targets = []
        for img_result in results:
            labels = img_result['labels']
            img_cls_result = {
                'boxes':[],
                'confidence':[]
            }
            for idx, label in enumerate(labels):
                if label == cls:
                    img_cls_result['boxes'].append(img_result['boxes'][idx])
                    img_cls_result['confidence'].append(img_result['scores'][idx])
            cls_results.append(img_cls_result)
        for img_target in targets:
            labels = img_target['labels']
            img_cls_target = {
                'boxes':[]
            }
            for idx, label in enumerate(labels):
                if label == cls:
                    img_cls_target['boxes'].append(img_target['boxes'][idx])
            cls_targets.append(img_cls_target)
        cal_ap(cls_results, cls_targets, iou_thresh)

def cal_ap(results, targets, iou_thresh):
    if len(results) != 0:
        print(results)
        print(targets)
    for result, target in zip(results, targets):
        pass