import os
import random
import argparse
import collections
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from synthesize.utils import *
from validation.utils import ImageFolder, RDED_ImageFolder









def load_model_3(model_name="resnet18", dataset="cifar10", pretrained=True, classes=[]):
    def get_model(model_name="resnet18"):
        if "conv" in model_name:
            if dataset in ["cifar10", "cifar100"]:
                size = 32
            elif dataset == "tinyimagenet":
                size = 64
            elif dataset in ["imagenet-nette", "imagenet-woof", "imagenet-100"]:
                size = 128
            else:
                size = 224

            nclass = len(classes)

            model = ConvNet(
                num_classes=nclass,
                net_norm="batch",
                net_act="relu",
                net_pooling="avgpooling",
                net_depth=int(model_name[-1]),
                net_width=128,
                channel=3,
                im_size=(size, size),
            )
        elif model_name == "resnet18_modified":
            model = thmodels.__dict__["resnet18"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        elif model_name == "resnet101_modified":
            model = thmodels.__dict__["resnet101"](pretrained=False)
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            model.maxpool = nn.Identity()
        else:
            model = thmodels.__dict__[model_name](pretrained=False)

        return model

    def pruning_classifier(model=None, classes=[]):
        try:
            model_named_parameters = [name for name, x in model.named_parameters()]
            for name, x in model.named_parameters():
                if (
                    name == model_named_parameters[-1]
                    or name == model_named_parameters[-2]
                ):
                    x.data = x[classes]
        except:
            print("ERROR in changing the number of classes.")

        return model

    # "imagenet-100" "imagenet-10" "imagenet-first" "imagenet-nette" "imagenet-woof"
    model = get_model(model_name)
    model = pruning_classifier(model, classes)

    # 특정 경로에서 모델 로드
    if pretrained:
        model_path = "/home/sb/link/DD_DIF/exp/cifar100_conv3_f1_mipc100_ipc10_cr5_Guided_F/saved_models/final_model.pth"
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # state_dict 키 확인 및 처리
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        
        # 수정된 state_dict 로드
        model.load_state_dict(new_state_dict)
        
        print(f"Model loaded from {model_path}")
    else:
        print("Model initialized without pre-training")

    return model










################################################### Original RDED Codes ###################################################
def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir

def init_images(args, model=None):
    additional_path = 'Diffusion_image'
    parent_dir = remove_last_directory(args.syn_data_path)
    Dif_Dir = os.path.join(parent_dir, additional_path)
    additional_path_2 = 'syn_data'
    RDED_Dir = os.path.join(parent_dir, additional_path_2)
    paths = [Dif_Dir]
    # paths = [Dif_Dir, RDED_Dir]


    trainset = RDED_ImageFolder(
        classes=args.classes,
        ipc=args.mipc,
        paths=paths, #train_dir
        shuffle=True,
        transform=None,
    )

    trainset.transform = transforms.Compose(
        [
            transforms.ToTensor(),
            MultiRandomCrop(
                num_crop=args.num_crop, size=args.input_size, factor=args.factor
            ),
            normalize,
        ]
    )
    print(args.factor)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        # print(len(train_loader))
        # print("Class", c)
        images = selector(
            args.ipc * args.factor**2,
            model,
            images,
            labels,
            args.input_size,
            m=args.num_crop,
        )
        images = mix_images(images, args.input_size, args.factor, args.ipc)
        save_images(args, denormalize(images), c)


def save_images(args, images, class_id):
    additional_path = 'RDED_after_Dif'

    parent_dir = remove_last_directory(args.syn_data_path)
    save_path = os.path.join(parent_dir, additional_path)
    # save_path=os.path.join(args.syn_data_path, additional_path)
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(save_path, class_id)
        place_to_store = dir_path + "/class{:05d}_id{:05d}.jpg".format(class_id, id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)


def main(args):
    print(args)
    additional_path = 'RDED_after_Dif'
    save_path=os.path.join(args.syn_data_path, additional_path)
    with torch.no_grad():
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            shutil.rmtree(save_path)
            os.makedirs(save_path)

        model_teacher = load_model_3(
            model_name=args.arch_name,
            dataset=args.subset,
            pretrained=True,
            classes=args.classes,
        )

        model_teacher = nn.DataParallel(model_teacher).cuda()
        model_teacher.eval()
        for p in model_teacher.parameters():
            p.requires_grad = False

        init_images(args, model_teacher)


if __name__ == "__main__":
    pass