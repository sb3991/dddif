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

def remove_last_directory(path):
    parent_dir, _ = os.path.split(path)
    return parent_dir

def init_images(args, dir_path, source_name, model=None, use_selector_2=False, num_images=None, selected_images=None):
    trainset = RDED_ImageFolder(
        classes=args.classes,
        ipc=args.mipc,
        paths=[dir_path],
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
    # print("args.factor", args.factor)


    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.mipc,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    for c, (images, labels) in enumerate(tqdm(train_loader)):
        if use_selector_2:
            images = selector_2(
                num_images * args.factor**2,
                model,
                images,
                labels,
                args.input_size,
                selected_images,
                m=args.num_crop,
            )
        else:
            images = selector(
                num_images * args.factor**2,
                model,
                images,
                labels,
                args.input_size,
                m=args.num_crop,
            )
        images = mix_images(images, args.input_size, args.factor, num_images)
        save_images(args, denormalize(images), c, source_name)
    
    return images


def save_images(args, images, class_id, source_name):
    additional_path = 'RDED_after_Dif'
    parent_dir = remove_last_directory(args.syn_data_path)
    save_path = os.path.join(parent_dir, additional_path)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for id in range(images.shape[0]):
        dir_path = "{}/{:05d}".format(save_path, class_id)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        place_to_store = dir_path + "/class{:05d}_id{:05d}_{}.jpg".format(class_id, id, source_name)
        image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
        pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
        pil_image.save(place_to_store)

def main(args):
    print(args)
    
    model_teacher = load_model(
        model_name=args.arch_name,
        dataset=args.subset,
        pretrained=True,
        classes=args.classes,
    )

    model_teacher = nn.DataParallel(model_teacher).cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    parent_dir = remove_last_directory(args.syn_data_path)

    # Clear the target directory if it exists
    save_path = os.path.join(parent_dir, 'RDED_after_Dif')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # Get the number of images to select from each directory
    dif_images = 10 #args.dif_images if hasattr(args, 'dif_images') else args.ipc // 2
    rded_images = args.ipc - dif_images

    # Process Diffusion images using selector
    Dif_Dir = os.path.join(parent_dir, 'Diffusion_image')
    selected_images = process_directory(args, Dif_Dir, 'Dif', model_teacher, use_selector_2=False, num_images=dif_images)
    print(f'Saving {dif_images} low loss images with Dif')

    # Process RDED images using selector_2
    RDED_Dir = os.path.join(parent_dir, 'syn_data')
    process_directory(args, RDED_Dir, 'RDED', model_teacher, use_selector_2=True, num_images=rded_images, selected_images=selected_images)
    print(f'Saving {rded_images} least similar images with OG')

def process_directory(args, dir_path, source_name, model, use_selector_2=False, num_images=None, selected_images=None):
    print(f"Processing images from: {dir_path}")
    with torch.no_grad():
        return init_images(args, dir_path, source_name, model, use_selector_2, num_images, selected_images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dif_images', type=int, default=None, help='Number of images to select from Dif_Dir')
    # Add other arguments as needed
    args = parser.parse_args()
    
    if args.dif_images is None:
        args.dif_images = args.ipc // 2
    
    main(args)