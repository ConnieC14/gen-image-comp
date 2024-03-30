import sys
import numpy as np
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import clip
from test_bench_dataset import COCOImageDataset
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_dir', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

opt = parser.parse_args()
args={}
test_dataset=COCOImageDataset(test_bench_dir='test_bench', result_dir=opt.result_dir)

# TODO: Edit to compare one image
test_img = Image.open('examples/image/example_1.png').convert("RGB")
ref_img = Image.open('examples/image/reference/example_1.png')
clip_model,preprocess = clip.load("ViT-B/32", device="cuda")                
sum=0
count=0

result_feat = clip_model.encode_image(test_img)
ref_feat = clip_model.encode_image(ref_img)
result_feat=result_feat.to('cpu')
ref_feat=ref_feat.to('cpu')
result_feat = result_feat / result_feat.norm(dim=-1, keepdim=True)
ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
similarity = (100.0 * result_feat @ ref_feat.T)
sum=sum+similarity.item()
count=count+1

print(sum/count)