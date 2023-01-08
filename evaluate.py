import numpy as np
import torch

def eval_voc(results, targets, iou_thresh, num_classes):
    for result in results:
        result["boxes"] = result["boxes"].cpu().clone().detach().numpy()
        result["labels"] = result["labels"].cpu().clone().detach().numpy()
        result["scores"] = result["scores"].cpu().clone().detach().numpy()
    for target in targets:
        target["boxes"] = target["boxes"].cpu().clone().detach().numpy()
        target["labels"] = target["labels"].cpu().clone().detach().numpy()
        
    ap_per_cls = []
    ap_sum = .0
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
        ap = cal_ap(cls_results, cls_targets, iou_thresh)
        ap_per_cls.append(ap)
        ap_sum += ap
    mAP = ap_sum / num_classes
    print("mAP = {:.4f}".format(mAP))
    return mAP, ap_per_cls
    
        
def cal_ap(results, targets, iou_thresh):
    gt_count = 0
    tp = np.array([],dtype=int)
    fp = np.array([],dtype=int)
    conf = np.array([])
    for result, target in zip(results, targets):
        gt_boxes = np.array(target['boxes'])
        pred_boxes = np.array(result['boxes'])

        gt_count += len(gt_boxes)
        if len(pred_boxes) == 0:
            continue
        img_tp = np.zeros(len(pred_boxes))
        img_fp = np.ones(len(pred_boxes))
        if len(gt_boxes) != 0:
            match_matrix = box_iou(gt_boxes, pred_boxes)
            maxidx = match_matrix.argmax(axis=1)
            for idx, match in zip(maxidx, match_matrix):
                maxIou = match[idx]
                if maxIou > iou_thresh:
                    img_tp[idx] = 1
                    img_fp[idx] = 0
        tp = np.concatenate((tp, img_tp))
        fp = np.concatenate((fp, img_fp))
        conf = np.concatenate((conf, result['confidence']))
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / float(gt_count)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = cal_voc_ap(rec, prec)

    return ap

def cal_voc_ap(rec, prec):
    recall = np.concatenate(([0.], rec, [1.0]))
    precision = np.concatenate(([0.], prec, [0.]))
    
    for i in range(len(precision)-1, 0, -1):
        precision[i-1] = np.maximum(precision[i-1], precision[i])
    
    i = np.where(recall[1:] != recall[:-1])[0]
    
    ap = np.sum((recall[i+1] - recall[i]) * precision[i+1])        

    return ap

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    @param boxes1: boxes one, shape: (N, 4)
    @param boxes2: boxes two, shape: (M, 4)
    @return:
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = rb - lt
    wh = np.clip(wh, 0, np.Inf)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


if __name__ == "__main__":
    pass