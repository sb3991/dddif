import os
import csv
import random
import warnings
import argparse

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import logging
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from synthesize.utils import load_model
from validation.utils import (
    ImageFolder,
    ShufflePatches,
    mix_aug,
    AverageMeter,
    accuracy,
    get_parameters,
)
#RDED Original Validation Codes

import torch
import torch.nn.functional as F

def label_smoothing(hard_label, num_classes, alpha):
    """
    Apply label smoothing to the hard labels.
    """
    # Convert hard_label (class indices) to one-hot encoding
    hard_label = F.one_hot(hard_label, num_classes).float()
    smoothed_labels = hard_label * (1 - alpha) + alpha / num_classes
    return smoothed_labels


def GIFT(num_classes, pred_label, soft_label, hard_label, alpha, weight ,temperature):
    #원래 들어와야 하는 것은 OUTPUT이랑 SOFT LABEL, 그리고 HARD LABEL과 PARAMETER
    smooth_label = label_smoothing(hard_label, num_classes, alpha)
    # soft_label = F.softmax(soft_label, dim=1)
    soft_label = F.normalize(soft_label, dim=1)
    smooth_label = F.normalize(smooth_label, dim=1)
    # print("soft_label", soft_label[0])
    # print("smooth_label", smooth_label[0])
    refined_soft_labels = weight * smooth_label + (1 - weight) * soft_label


    # CE Loss   
    # loss = torch.mean(torch.sum(-refined_soft_labels * F.log_softmax(pred_label, dim=-1), dim=-1))


    # KL Divergence

    # loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    # pred_label= F.log_softmax(pred_label/ temperature, dim=1)
    # refined_soft_labels=F.softmax(soft_label / temperature, dim=1) #refined_soft_labels
    # loss =loss_function_kl(pred_label, refined_soft_labels)

    # Cosine Sim

    # pred_label= F.softmax(pred_label/ temperature, dim=1)
    # refined_soft_labels=F.softmax(refined_soft_labels / temperature, dim=1) #refined_soft_labels
    loss = -F.cosine_similarity(pred_label, refined_soft_labels, dim=1).mean()
    # loss = -loss.mean()  # Use the negative mean cosine similarity as the loss
    return loss


sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)




def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    main_worker(args)

def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir

def main_worker(args):
    additional_path = 'RDED_after_Dif'
    parent_dir = remove_last_directory(args.syn_data_path)
    save_path = os.path.join(parent_dir, additional_path)
    # save_path=os.path.join(args.syn_data_path, additional_path)
    # original_path="/home/sb/link/DD_DIF/exp/tinyimagenet_resnet18_modified_f1_mipc300_ipc10_cr5/syn_data/OG_RDED"
    # save_path="/home/a5t11/ICML_Symbolic/Repo/DD/RDED/exp/cifar100_conv3_f1_mipc300_ipc10_cr5/syn_data"
    print("=> using pytorch pre-trained teacher model '{}'".format(args.arch_name))
    teacher_model = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )

    student_model = load_model(
        model_name=args.stud_name,
        dataset=args.subset,
        pretrained=False,
        classes=args.classes,
    )
    teacher_model = torch.nn.DataParallel(teacher_model).cuda()
    student_model = torch.nn.DataParallel(student_model).cuda()

    teacher_model.eval()
    student_model.train()

    # freeze all layers
    for param in teacher_model.parameters():
        param.requires_grad = False

    cudnn.benchmark = True

    # optimizer
    if args.sgd:
        optimizer = torch.optim.SGD(
            get_parameters(student_model),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            get_parameters(student_model),
            lr=args.adamw_lr,
            betas=[0.9, 0.999],
            weight_decay=args.adamw_weight_decay,
        )

    # lr scheduler
    if args.cos == True:
        scheduler = LambdaLR(
            optimizer,
            lambda step: 0.5 * (1.0 + math.cos(math.pi * step / args.re_epochs / 2))
            if step <= args.re_epochs
            else 0,
            last_epoch=-1,
        )
    else:
        scheduler = LambdaLR(
            optimizer,
            lambda step: (1.0 - step / args.re_epochs) if step <= args.re_epochs else 0,
            last_epoch=-1,
        )

    print("process data from {}".format(save_path))
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    augment = []
    augment.append(transforms.ToTensor())
    augment.append(ShufflePatches(args.factor))
    augment.append(
        transforms.RandomResizedCrop(
            size=args.input_size,
            scale=(1 / args.factor, args.max_scale_crops),
            antialias=True,
        )
    )
    augment.append(transforms.RandomHorizontalFlip())
    augment.append(normalize)

    train_dataset = ImageFolder(
        classes=range(args.nclass),
        ipc=args.ipc,
        mem=True,
        shuffle=True,
        root=save_path,
        transform=transforms.Compose(augment),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.re_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )

    val_loader = torch.utils.data.DataLoader(
        ImageFolder(
            classes=args.classes,
            ipc=args.val_ipc,
            mem=True,
            root=args.val_dir,
            transform=transforms.Compose(
                [
                    transforms.Resize(args.input_size // 7 * 8, antialias=True),
                    transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.re_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=set_worker_sharing_strategy,
    )
    print("load data successfully")

    best_acc1 = 0
    best_epoch = 0
    args.optimizer = optimizer
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader
    # args.additional_loader = additional_loader

    for epoch in range(args.re_epochs):
        train(epoch, train_loader, teacher_model, student_model, args)

        if epoch % 50 == 9 or epoch == args.re_epochs - 1:
            if epoch > args.re_epochs * 0.4:
                top1 = validate(student_model, args, epoch)
            else:
                top1 = 0
        else:
            top1 = 0

        scheduler.step()
        if top1 > best_acc1:
            best_acc1 = max(top1, best_acc1)
            best_epoch = epoch


    last_epoch_acc = top1  # 마지막 에포크의 정확도 저장

    print(f"Train Finish! Best accuracy is {best_acc1}@{best_epoch}")

    # CSV 파일로 결과 저장
    csv_filename = os.path.join(parent_dir, 'training_results.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value', 'Epoch'])
        writer.writerow(['Best Accuracy', best_acc1, best_epoch])
        writer.writerow(['Last Epoch Accuracy', last_epoch_acc, args.re_epochs - 1])

    print(f"Results saved to {csv_filename}")


def train(epoch, train_loader, teacher_model, student_model, args):
    """Generate soft labels and train"""
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    optimizer = args.optimizer
    loss_function_kl = nn.KLDivLoss(reduction="batchmean")
    loss_function_ce = nn.CrossEntropyLoss()
    teacher_model.eval()
    student_model.train()
    t1 = time.time()

    total_correct_teacher = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()

            mix_images, _, _, _ = mix_aug(images, args)
            pred_label = student_model(images)
            num_classes = pred_label.size(1)  # Assuming the output shape is [batch_size, num_classes]

            soft_mix_label_0 = teacher_model(mix_images)
            soft_mix_label = F.softmax(soft_mix_label_0 / args.temperature, dim=1)

            predicted_classes_teacher = soft_mix_label_0.argmax(dim=1)
            correct_predictions_teacher = predicted_classes_teacher.eq(labels).sum().item()

            # Accumulate the total correct predictions and total samples for teacher model
            total_correct_teacher += correct_predictions_teacher
            total_samples += labels.size(0)

        if batch_idx % args.re_accum_steps == 0:
            optimizer.zero_grad()

        prec1, prec5 = accuracy(pred_label, labels, topk=(1, 5))

        pred_mix_label = student_model(mix_images)

        soft_pred_mix_label = F.log_softmax(pred_mix_label / args.temperature, dim=1)
        # loss= GIFT(num_classes, pred_mix_label, soft_mix_label_0, labels, 0.1, 0.1, args.temperature)
        loss = loss_function_kl(soft_pred_mix_label, soft_mix_label)
        # loss = loss_function_ce(pred_mix_label, labels)
        loss = loss / args.re_accum_steps

        loss.backward()
        if batch_idx % args.re_accum_steps == (args.re_accum_steps - 1):
            optimizer.step()

        n = images.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

    # Calculate and print the teacher model's accuracy for the epoch
    if epoch % 10 ==9:
        teacher_accuracy = total_correct_teacher / total_samples * 100
        print(f"Epoch {epoch}: Teacher Model Accuracy: {teacher_accuracy:.2f}%")

    printInfo = (
        "TRAIN Epoch {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "train_time = {:.6f}".format((time.time() - t1))
    )
    print(printInfo)
    t1 = time.time()


def validate(model, args, epoch=None):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_function = nn.CrossEntropyLoss()

    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data, target in args.val_loader:
            target = target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = (
        "TEST:\nIter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    print(logInfo)
    logInfo = (
        "TEST: Iter {}: loss = {:.6f},\t".format(epoch, objs.avg)
        + "Top-1 err = {:.6f},\t".format(100 - top1.avg)
        + "Top-5 err = {:.6f},\t".format(100 - top5.avg)
        + "val_time = {:.6f}".format(time.time() - t1)
    )
    return top1.avg


if __name__ == "__main__":
    pass
    # main(args)
