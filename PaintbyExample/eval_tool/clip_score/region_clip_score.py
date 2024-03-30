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
import csv

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_dir', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

opt = parser.parse_args()
args={}
test_dataset=COCOImageDataset(test_bench_dir='test_bench', result_dir=opt.result_dir)

test_dataloader= torch.utils.data.DataLoader(test_dataset, 
                                    batch_size=1, 
                                    num_workers=0, # JRF - I get errors when > 0
                                    pin_memory=True, 
                                    shuffle=False,#sampler=train_sampler, 
                                    drop_last=True)
clip_model,preprocess = clip.load("ViT-B/32", device="cpu")                
sum=0
count=0
results=dict()
for crop_tensor,ref_image_tensor in tqdm(test_dataloader):
    crop_tensor=crop_tensor.to('cpu')
    ref_image_tensor=ref_image_tensor.to('cpu')
    result_feat = clip_model.encode_image(crop_tensor)
    ref_feat = clip_model.encode_image(ref_image_tensor)
    result_feat=result_feat.to('cpu')
    ref_feat=ref_feat.to('cpu')
    result_feat = result_feat / result_feat.norm(dim=-1, keepdim=True)
    ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
    similarity = (100.0 * result_feat @ ref_feat.T)

    img_id = str(count+1).zfill(12)
    results[img_id] = similarity.item()
    print("Similarity [%s]: %s" % (img_id, similarity.item()))

    sum=sum+similarity.item()
    count=count+1

print("Average Similarity: %s" % (sum/count))

filename = "CLIP_output.csv"

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    writer.writerow(["img_id", "CLIP_score"])
    
    for key, value in results.items():
        writer.writerow([key, value])

print(f"CLIP results saved to {filename}")