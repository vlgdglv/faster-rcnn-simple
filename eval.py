import math
import numpy as np
import torch
import torchvision


def process_result(results, targets, iou_thresh, num_classes):
    cnt_results = 0
    cnt_targets = 0
    final_results = []
    # detach, numpify every thing:
    for result in results:
        result["boxes"] = result["boxes"].cpu().clone().detach().numpy()
        result["labels"] = result["labels"].cpu().clone().detach().numpy()
        result["scores"] = result["scores"].cpu().clone().detach().numpy()
        num_result = result["boxes"].shape[0]
        final_results.append({
            "labels": result["labels"].copy(),
            "scores": result["scores"].copy(),
            "TP": np.zeros(num_result),
        })
        cnt_results += num_result

    target_records = []
    for target in targets:
        target["boxes"] = target["boxes"].cpu().clone().detach().numpy()
        target["labels"] = target["labels"].cpu().clone().detach().numpy()
        num_target = target["boxes"].shape[0]
        target_records.append({
            "matched_result_idx": np.full(num_target, -1),
            "best_matched_iou": np.full(num_target, -1)
        })
        cnt_targets += num_target
    gt_per_class = np.zeros(num_classes + 1)

    # print(result[0])
    # print(result[1])
    print("total results: {}, total targets: {}, iou threshold: {:.2f}".format(cnt_results, cnt_targets, iou_thresh))
    # process ecah images
    for image_idx, (result, target) in enumerate(zip(results, targets)):
        assert result["boxes"].shape[0] == result["labels"].shape[0] == result["scores"].shape[0]
        gt_labels = target["labels"]
        gt_boxes = target["boxes"]
        pred_labels = result["labels"]
        pred_boxes = result["boxes"]
        pred_scores = result["scores"]
        # will be used in calculating recall
        for label in gt_labels:
            gt_per_class[label] += 1
        if result["boxes"].shape[0] == 0:
            continue
        
        # process each prediction result
        for pred_idx, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            # step 1: select gt index of predicted label, and extract their gt boxes
            matched_class_idx = np.where(label == gt_labels)[0]
            class_gt_boxes = gt_boxes[matched_class_idx]

            # if no such label, break
            if class_gt_boxes.shape[0] == 0:
                continue
            # step 2: calc ious of each gt boxes
            match_quality = box_iou(box[None, :], class_gt_boxes)[0]
            # sort by descend order and get idx
            sorted_idx = np.argsort(-match_quality)

            for i in sorted_idx:
                # get corresponding gt index
                matched_gt_idx = matched_class_idx[i]
                # ious
                match_score = match_quality[i]
                if match_score < iou_thresh:
                    # if current score is below threshold, then following score is too, no need to proceed
                    break
                # if no previous prediction matched current gt, then assign this prediction to current gt
                if target_records[image_idx]["matched_result_idx"][matched_gt_idx] == -1:
                    target_records[image_idx]["matched_result_idx"][matched_gt_idx] = pred_idx
                    target_records[image_idx]["best_matched_iou"][matched_gt_idx] = match_score
                    final_results[image_idx]["TP"][pred_idx] = 1
                    break
                else:
                    # some predictions has been assigned to current gt
                    temp_iou = target_records[image_idx]["best_matched_iou"][matched_gt_idx]
                    # but current prediction has higher iou, then replace with current prediction
                    if match_score > temp_iou:
                        temp_match_idx = target_records[image_idx]["matched_result_idx"][matched_gt_idx]
                        target_records[image_idx]["matched_result_idx"][matched_gt_idx] = pred_idx
                        final_results[image_idx]["TP"][temp_match_idx] = 0
                        final_results[image_idx]["TP"][pred_idx] = 1
                        target_records[image_idx]["best_matched_iou"][matched_gt_idx] = match_score
                        break

    return final_results, gt_per_class


def cal_mAP(results, gt_count, num_classes):
    """
    @param results: List[Dict[Str:List]]
    @return:
    """
    result_per_class = [{"scores": [], "TP":[]} for i in range(num_classes+1)]
    for result in results:
        labels = result["labels"]
        scores = result["scores"]
        TP = result["TP"]
        for label, score, tp in zip(labels, scores, TP):
            result_per_class[label]["scores"].append(score)
            result_per_class[label]["TP"].append(tp)

    ap_per_class = []
    tps = []
    ap_sum = .0
    for idx, result_dict in enumerate(result_per_class):
        ap, tp_count = cal_classAP(result_dict, gt_count[idx])
        ap_per_class.append(ap)
        tps.append(tp_count)
        ap_sum += ap
        
    print("tps:\t", end='')
    for tp in tps:
        print("{:d}\t".format(int(tp)), end='')
    print(' ')
    print("gts:\t", end='')
    s = 0
    for gt in gt_count:
        s += gt
        print("{:d}\t".format(int(gt)), end='') 
    print("{:d}\t".format(int(s)), end='')
    print(' ')
    mAP = ap_sum / num_classes
    
    return mAP, ap_per_class


def cal_classAP(result, recall_total):
    if recall_total == 0:
        return .0, 0
    scores = result["scores"]
    TPs = result["TP"]
    sorted_idx = np.argsort(-np.array(scores))
    total_sum = 0
    tp_count = 0
    precision = []
    recall = []
    for idx in sorted_idx:
        total_sum += 1
        if TPs[idx] == 1:
            tp_count += 1
            precision.append(tp_count / total_sum)
            recall.append(tp_count / recall_total)
    p_sum = .0
    for idx in range(len(precision)):
        p_sum += np.max(precision[idx:])
    ap = p_sum / recall_total
    return ap, tp_count


def eval_mAP(results, targets, iou_thresh, num_classes):
    gt_count = np.zeros(num_classes + 1)
    result, batch_gt_count = process_result(results, targets, iou_thresh, num_classes)
    for idx, num in enumerate(batch_gt_count):
        gt_count[idx] += num
    return cal_mAP(result, gt_count, num_classes)


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
    targets = [{'boxes': torch.tensor([[4, 5, 459, 500]], device='cuda:0'),
                'labels': torch.tensor([12], device='cuda:0')},
                {'boxes': torch.tensor([[389, 131, 433, 292],
                                        [362, 190, 393, 257],
                                        [3, 0, 117, 48]], device='cuda:0'),
                 'labels': torch.tensor([15, 7, 15], device='cuda:0')},
                {'boxes': torch.tensor([[329, 3, 493, 333],
                                        [448, 1, 500, 131],
                                        [3, 0, 107, 28]], device='cuda:0'),
                 'labels': torch.tensor([15, 1, 15], device='cuda:0')},
                {'boxes': torch.tensor([[16, 5, 287, 500]], device='cuda:0'),
                 'labels': torch.tensor([14], device='cuda:0')}]

    results = [{'boxes': torch.tensor([], device='cuda:0', dtype=torch.int32),
                'labels': torch.tensor([], device='cuda:0', dtype=torch.int64),
                'scores': torch.tensor([], device='cuda:0')},
               {'boxes': torch.tensor([[2, 0, 102, 58], [3, 0, 107, 28]], device='cuda:0', dtype=torch.int32),
               'labels': torch.tensor([19, 15], device='cuda:0'),
                'scores': torch.tensor([0.0504, 0.0502], device='cuda:0')},
               {'boxes': torch.tensor([[30, 10, 207, 48],
                                       [319, 10, 500, 320]], device='cuda:0', dtype=torch.int32),
                'labels': torch.tensor([9, 15], device='cuda:0', dtype=torch.int64),
                'scores': torch.tensor([0.5, 0.8], device='cuda:0')},
               {'boxes': torch.tensor([[0, 1, 35, 34],
                                       [2, 0, 251, 156],
                                       [1, 0, 71, 35]], device='cuda:0', dtype=torch.int32),
                'labels': torch.tensor([2, 14, 19], device='cuda:0'),
                'scores': torch.tensor([0.0501, 0.0500, 0.0500], device='cuda:0')}]

    # num_classes = 20
    # gt_count = np.zeros(num_classes + 1)
    # result1, batch_gt_count = process_result(results, targets, 0.5, num_classes)
    #
    # for idx, num in enumerate(batch_gt_count):
    #     gt_count[idx] += num
    # mAP, ap_per_class = cal_mAP(result1, gt_count, num_classes)
    # print(mAP)

    fake_result = {
        "scores": [.23, .76, .01, .91, .13, .45, .12, .03, .38, .11, .03, .09, .65, .07, .12, .24, .10, .23, .46, .08],
        "TP": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    }
    # cal_classAP(fake_result, 6)
