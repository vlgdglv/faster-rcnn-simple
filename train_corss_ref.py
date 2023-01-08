import os
import math
import time
import torch
import argparse
import torchvision
import wandb
import datetime as dt
import numpy as np
import pickle as pk
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from VOCDataset import VOCDataset, VOCLoader
from eval import eval_mAP
from evaluate import eval_voc

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

classes =  [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--lr', default=0.02, type=float, help='inital learning rate')
    parser.add_argument('--scheduler', type=str, choices=['step', 'cos'], default='', help='optimizer')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='SGD', help='optimizer')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--num-classes', default=20, type=str, help='number of class')
    parser.add_argument('--eval-iou-thresh', default=0.5, type=float, help='pascal voc mAP iou threshold')
    parser.add_argument('--wandb', default=False, type=bool, help='enable wandb')
    parser.add_argument('--debug', default=False, type=bool, help='debug mode, only 10 iterations for train and val')
    parser.add_argument('--weights', default=None, type=str, help='load weights')
    return parser.parse_args()


def main(opt):
    # only for debug
    if opt.wandb:
        wandb.init(project="faster_rcnn", 
                entity="vlgunsdaddy",
                name=dt.datetime.strftime(dt.datetime.now(), 'train-0 %Y-%m-%d %H:%M:%S'))

    epochs = opt.epochs
    batch_size = opt.batch_size
    num_classes = opt.num_classes
    eval_iou_thresh = opt.eval_iou_thresh
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join("./runs", opt.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # prepare dataset
    voc2012 = VOCDataset("./dataset/VOC2012")
    vocloader = VOCLoader(dataset=voc2012, batch_size=batch_size, shuffle=True, val_proportion=0.3)

    # load models
    backbone = torchvision.models.vgg16(pretrained=True).features
    backbone.out_channels = 512
    # model = FasterRCNN(backbone, num_classes=num_classes + 1, min_size=500, max_size=800)
    
    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone, rpn_anchor_generator=anchor_generator, num_classes=num_classes+1, min_size=500, max_size=600)
    
    #load weights
    if opt.weights:
        model.load_state_dict(torch.load(opt.weights))
    
    model.to(device)

    # learing rate
    lr = opt.lr
    # optimizer
    optim_type = opt.optimizer
    if optim_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # scheduler
    scheduler_type = opt.scheduler
    if scheduler_type == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None
    
    if opt.wandb:
        wandb.config = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size
        }

    print("[INFO] training begin")
    print("epochs: {}, batch size: {}".format(epochs, batch_size))

    mAP_best = -1
    num_training_samples = vocloader.num_train
    for epoch in range(epochs):
        print("Epoch {}: training, learning rate={}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        vocloader.train()
        loss_sum = 0
        for i, (images, targets) in tqdm(enumerate(vocloader)):
            images = [image.to(device)  for image in images]
            for target in targets:
                target["boxes"] = target["boxes"].to(device)
                target["labels"] = target["labels"].to(device)

            losses = model(images, targets)
            
            loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"] + \
                   losses["loss_box_reg"] + losses["loss_classifier"]
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.debug:
                if i==5:
                    break    
        if scheduler is not None:
            scheduler.step()
        if opt.wandb:
            wandb.log({"loss": loss_sum / num_training_samples})
        print("Epoch {}: loss = {:.4f}".format(epoch, loss_sum / num_training_samples))
        print("Epoch {}: validating".format(epoch))
        model.eval()
        vocloader.val()
        results_list = []
        targets_list = []
        with torch.no_grad():
            for i, (images, targets) in tqdm(enumerate(vocloader)):
                images = [image.to(device)  for image in images]
                for target in targets:
                    target["boxes"] = target["boxes"].to(device)
                    target["labels"] = target["labels"].to(device)

                results = model(images, targets)
                for result in results:
                    results_list.append(result)
                for target in targets:
                    targets_list.append(target)
                if opt.debug:
                    if i==5:
                        break
                
            mAP, ap_per_class = eval_voc(results_list, targets_list, eval_iou_thresh, num_classes)
            with open(os.path.join(save_dir, 'AP_per_class.pkl'), 'wb') as f:
                pk.dump(ap_per_class, f)
            print("epoch {}, mAP={:.3f}".format(epoch, mAP))
            print("ap\t", end="")
            for ap in ap_per_class:
                print("{:.2f}\t".format(ap), end='')
            print(" ")
            print("class\t", end="")
            for i in range(len(ap_per_class)):
                print("{}\t".format(classes[i][:3]), end='')
            print(" ")
        if opt.wandb:
            wandb.log({"mAP": mAP})
            cls_dict ={}
            for idx, (cls, ap)  in enumerate(zip(classes, ap_per_class)):
                cls_dict["{}:{}".format(idx, cls)] = ap
            wandb.log(cls_dict)
        if mAP > mAP_best:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'last.pt'))


if __name__ == "__main__":
    main(parser_opt())
