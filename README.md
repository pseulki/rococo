# RoCOCO: Robustness Benchmark of MS-COCO to Stress-test Image-Text Matching Models
[RoCOCO: Robustness Benchmark of MS-COCO to Stress-test Image-Text Matching Models](https://arxiv.org/abs/2304.10727) (ECCV Synthetic Data for Computer Vision Workshop 2024 Oral).
<p align="center">
<img src="https://github.com/pseulki/rococo/blob/main/blip_example2.png" width=75% height=75% align="center">
</p>

[Slides](https://docs.google.com/presentation/d/1XMvLpHNT3JU2dW1go-SJ0WmdywlKl2YG/edit?usp=sharing&ouid=109428068995513711293&rtpof=true&sd=true) | [Bibtex](#Citation) 
## 1. Data Preparation
COCO dataset can be downloaded from [here](https://cocodataset.org/#home).

Altered captions (Same-concept, Diff-concept, Rand-voca, Danger) are located in the 'annotations' directory. 
Json file include the image path, ground-truth captions, and adversarial captions. 
 ```bash
 .
├── annotations                 
├   ├── rand_voca.json                       
├   ├── diff_concept.json             
├   ├── same_concept.json           
├   ├── danger.json      
```

To generate the altered images, use the code provided below. 
For models utilizing BUTD features, we offer precomputed BUTD features for the altered images with random seeds 1, 10, 100 at [this link](https://drive.google.com/drive/folders/1KqjbqemR0BnjjAqft7agDEKTr5WAppJt?usp=sharing). 


### Create visually corrupted images 
- Requirements:
```
torchvision
opencv-python 
```
- `mix_images.py`
	- `method`: str (default: `mix`): Method to mix images (choices: mix, patch).
	- `lam`: float (default: 0.9): Proportion of mixing images.
	- `data_path`: str (default: `None`): Directory where COCO images locate. New mixed images will be also stored in this directory with the directory name of `method + '_' + str(lam)` (e.g., mix_0.9). 
	- `img_list`: str (default: `annotations/coco_karpathy_test.json`): COCO Test image lists.
    - `seed` : int (default: 1): Random seed. In the paper, results are averaged from experiments with 3 different seeds, `1, 10, 100`.
- Usage: 
```bash
python mix_images.py --method mix --lam 0.9 --data_path /data/coco/images/ --seed 1
```



## 2. Evaluation 

Since the retrieval methods vary for each baseline, it is necessary to follow the specific baseline method when calculating similarity using text/image embeddings. Here, we provide examples of evaluation methods from [CLIP](https://arxiv.org/abs/2103.00020), [BLIP](https://github.com/salesforce/BLIP), and [VSEInfty](https://github.com/woodfrog/vse_infty.git).

### (1) CLIP example
The CLIP evaluation code has been modified from the original [CLIP repository](https://github.com/OpenAI/CLIP) and [VSEInfty repository](https://github.com/woodfrog/vse_infty.git).
- Requirements:
```
pip install git+https://github.com/openai/CLIP.git
```
 #### Image-to-Text
- `eval_clip_i2t.py`
    - `data_path` : COCO Image directory (e.g., `data/coco/images/`)
	- `ann_file`: str (default: `rand_voca.json`): Adversarial Test dataset to list texts and images.
	- `clip_model`: str (default: `ViT-B/32`): Clip model for evaluation 
- Usage: 
```bash
python eval_clip_i2t.py --ann_file rand_voca.json --clip_model ViT-B/32
```

#### Text-to-Image
- `eval_clip_t2i.py`
	- `data_path` : COCO Image directory (e.g., `data/coco/images/`)
	- `miximage`: str (default: `mix_0.9`): Additional images created above to confuse the models.
	- `ann_file`: str (default: `coco_karpathy_test.json`): COCO Test dataset to list texts and images.
	- `clip_model`: str (default: `ViT-B/32`): Clip model for evaluation 

- Usage: 
```bash
python eval_clip_i2t.py --miximage mix_0.9 --clip_model ViT-B/32
```


### (2) BLIP example
The BLIP evaluation code has been slightly modified from the original [BLIP code  repository](https://github.com/salesforce/BLIP).
- Requirements:
```
pyyaml
transformers==4.15.0
timm==0.4.12
fairscale==0.4.4
pycocoevalcap
```
 #### Image-to-Text
- `eval_blip_i2t.py`
	- `testfilename`: str (default: `rand_voca.json`): Adversarial Test dataset to list texts and images.
	- `config`: str (default: `blip/configs/retrieval_coco.yaml`): Config file for BLIP. Config file has :
	    - image_root: COCO Image directory (e.g., `data/coco/images/`)
		- ann_root: Annotation directory (e.g., `annotation/`)
- Usage: 
```bash
python -m torch.distributed.run --nproc_per_node=4 eval_blip_i2t.py --testfilename rand_voca.json
```

#### Text-to-Image
- `eval_blip_t2i.py`
	- `miximage`: str (default: `mix_0.9`): Additional images created above to confuse the models.
	- `testfilename`: str (default: `coco_karpathy_test.json`): COCO Test dataset to list texts and images.
	- `config`: str (default: `blip/configs/retrieval_coco.yaml`): Config file for BLIP. Config file has :
	    - image_root: COCO Image directory (e.g., `data/coco/images/`)
		- ann_root: Annotation directory (e.g., `annotation/`)
- Usage: 
```bash
python -m torch.distributed.run --nproc_per_node=4 eval_blip_t2i.py --miximage mix_0.9
```
 

### (3) VSEInfty example

To be updated


## Citation

If you find our paper and repo useful, please cite our paper

```
@article{park2023rococo,
  title={Rococo: Robustness benchmark of ms-coco to stress-test image-text matching models},
  author={Park, Seulki and Um, Daeho and Yoon, Hajung and Chun, Sanghyuk and Yun, Sangdoo},
  journal={arXiv preprint arXiv:2304.10727},
  year={2023}
}
```
