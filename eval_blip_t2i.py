'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from blip.models.blip_retrieval import blip_retrieval
import blip.utils as utils
from blip.utils import cosine_lr_schedule
from blip.data import create_dataset, create_sampler, create_loader

@torch.no_grad()
def evaluation(model, data_loader, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text  # all texts
    num_text = len(texts)
    print("Num text: ", num_text)
    text_bs = 256
    text_ids = []
    text_embeds = []
    text_atts = []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = []
    image_embeds = []

    for image, img_id, _ in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed,dim=-1)

        image_feats.append(image_feat.cpu())
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)


    sims_matrix = image_embeds @ text_embeds.t()
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):

        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1),
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2i[start+i,topk_idx] = score + topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_t2i.cpu().numpy()



@torch.no_grad()
def itm_eval(sims, txt2img):
    # Text->Images
    false_cnt = 0
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

def main(args, config):
    if args.distributed:
        utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    # TODO: Remove train codes.
    print("Creating retrieval dataset")
    filenames = {'val': 'coco_karpathy_val.json', 'test': args.testfilename}
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config, filenames=filenames,
                                                              miximage=args.miximage)
    print("test dataset: ", len(test_dataset))
    print("test images: ", len(test_dataset.image))

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None,None,None])


    #### Model ####
    print("Creating model")
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    print("Start Evaluation")
    start_time = time.time()

    score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)

    if utils.is_main_process():
        test_result = itm_eval(score_test_t2i, test_loader.dataset.txt2img)
        print(test_result)

        log_stats = {#**{f'val_{k}': v for k, v in val_result.items()},
                     **{f'test_{k}': v for k, v in test_result.items()},
                    }
        modelname = config['pretrained'].split('/')[2].split('.')[0]
        save_file = os.path.join(args.output_dir, modelname + "_evaluate.txt")
        with open(save_file,"a") as f:
            f.write(json.dumps(log_stats) + "\n")

        #dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='blip/configs/retrieval_coco.yaml')
    parser.add_argument('--output_dir', default='output/blip')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--testfilename', default='coco_karpathy_test.json', type=str)
    parser.add_argument('--miximage', default='mix_0.9', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
