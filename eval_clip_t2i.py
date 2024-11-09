# cuda10.2_torch1.8
# Also confirmed working on torch2.1 & cuda12.1

import os
import argparse
import logging
import time
import datetime
from lib import evaluation_clip
from lib import datasets 
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import json

import clip
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def main(args):
    device = torch.device(args.device)

    model, preprocess = clip.load(args.clip_model, args.device)

    if args.data_path is not None:
        opt.data_path = args.data_path

    #### Dataset ####
    print("Creating retrieval dataset")
    test_loader = datasets.get_test_loader(args.data_path, opt, args,
                                            args.miximage, args.batch_size, args.workers, preprocess)
    print("test image dataset: ", len(test_loader.dataset))



    print("Start testing")
    start_time = time.time()
    sim = evaluation_clip.evalrank(model, test_loader, device)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    test_result = itm_eval(sim, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
    print(test_result)

    with open(os.path.join(args.save_path, args.clip_model.replace('/', '_') + "log.txt"), "a") as f:
        f.write(json.dumps(test_result) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


@torch.no_grad()
def itm_eval(sims, txt2img, img2txt):
    # Text->Images
    false_cnt = 0
    sims = sims.transpose()
    ranks = np.zeros(sims.shape[0])
    for index, score in enumerate(sims):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]
        if inds[0] >= 5000:
            false_cnt += 1
    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    false_rate = 100 * false_cnt / len(ranks)
    ir_mean = (ir1 + ir5 + ir10) / 3

    eval_result = {'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'false_rate': false_rate}
    return eval_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='MSCOCO/images')
    parser.add_argument('--save_path', default='./clip')
    parser.add_argument('--ann_path', default='annotations/')
    parser.add_argument('--ann_file', default='coco_karpathy_test.json')
    parser.add_argument('--miximage', default='mix_0.9', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--workers', default=5, type=int)
    parser.add_argument('--clip_model', default='ViT-B/32', type=str,  choices=['ViT-B/32', 'ViT-B/16',
                    'ViT-L/14', 'ViT-L/14@336px', 'RN50', 'RN50x4', 'RN50x16', 'RN50x64'])
    opt = parser.parse_args()

    print("args: ", "miximage: ", opt.miximage, "data_path: ", opt.data_path, "save_path: ", opt.save_path,
          "ann_file: ", opt.ann_file)


    main(opt)
