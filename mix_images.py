import os
import torchvision
from torchvision.transforms.functional import InterpolationMode
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import argparse

import torchvision.transforms as transforms

from torchvision.utils import save_image
import cv2
import json

class coco_karpathy_caption_eval(Dataset):
    def __init__(self, transform, image_root, image_list):
        '''
        Code from: https://github.com/salesforce/BLIP
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.transform = transform
        self.image_root = image_root
        self.image = []

        self.annotation = json.load(open(image_list, 'r'))

        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, image_path

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='mix', choices=['mix', 'patch'], help='methods of mix images')
    parser.add_argument('--seed', default='1', type=int)
    parser.add_argument('--lam', default='0.9', type=float, help='proportion of mixing images')
    parser.add_argument('--data_path', default=None, help='data path of the coco images')
    parser.add_argument('--img_list', default='data/coco_karpathy_test.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    lam = args.lam
    method = args.method
    save_folder =args.data_path + method + '_' + str(lam) + '/'
    read_folder = args.data_path
    img_list = args.img_list

    print("save_folder", save_folder)
    if not os.path.exists(save_folder):
        print("create folder: ", save_folder)
        os.makedirs(save_folder)


    transform_test = transforms.Compose([
            transforms.Resize((384, 384),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    test_dataset= coco_karpathy_caption_eval(transform=transform_test, image_root=read_folder, image_list=img_list)
    batch_size = 64

    train_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)
    print("len", len(train_loader.dataset))
    cnt = 0
    for i, (input, filename) in enumerate(train_loader):
        rand_index = torch.randperm(input.size()[0])
        if method == 'patch':
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        elif method == 'mix':
            input = lam * input + (1 - lam) * input[rand_index, :]

        for j in range(input.size(0)):
            img_name = filename[j].split('/')[-1]
            title = os.path.join(save_folder, img_name)
            save_image(input[j], title)
            cnt += 1

    print("total images: ", cnt)


