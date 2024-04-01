from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import clip
import bezier



def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class COCOImageDataset(data.Dataset):
    def __init__(self,test_bench_dir,result_dir):

        self.test_bench_dir=test_bench_dir
        self.result_dir=result_dir
        self.id_list=np.load('test_bench/id_list.npy')
        self.id_list=self.id_list.tolist()
        # print("ID LIST: ",self.id_list)
        # print("length of test bench",len(self.id_list))
        # self.length=len(self.id_list)

        # print("Result Dir: ",self.result_dir)
        # print("Test Bench Dir: ",self.test_bench_dir)
        self.output_files = os.listdir('results/results')
        self.length = len(self.output_files)
       

    
    def __getitem__(self, index):
        result_img_name = self.output_files[index]
        parsed_result_name = result_img_name.split('_')
        mask_img_name, ref_img_name = parsed_result_name[0] + "_mask.png", parsed_result_name[1]
        print("Output File: ",result_img_name)
        print("Mask Image Name: ",mask_img_name)
        print("Ref Image Name: ",ref_img_name)

        # result_img_name = str(self.id_list[index]).zfill(12)+'.png'
        # print("Open results/%s" % str(self.id_list[index]).zfill(12)+'.png')

        result_path=os.path.join(os.path.join(self.result_dir,result_img_name))
        result_p = Image.open(result_path).convert("RGB")
        result_tensor = get_tensor_clip()(result_p)

        ### Get reference
        # print("Open Ref_3500/%s" % str(self.id_list[index]).zfill(12)+'_ref.png')
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'Ref_3500',ref_img_name))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_image_tensor=get_tensor_clip()(ref_img)

   
        ### bbox mask
        # print("Open Mask_bbox_3500/%s" % str(self.id_list[index]).zfill(12)+'_mask.png')
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'Mask_bbox_3500',mask_img_name))
        mask_img=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        idx0 = np.nonzero(mask_img.ravel()==255)[0]
        idxs = [idx0.min(), idx0.max()]
        out = np.column_stack(np.unravel_index(idxs,mask_img.shape))
        crop_tensor=result_tensor[:,out[0][0]:out[1][0],out[0][1]:out[1][1]]
        crop_tensor=T.Resize([224,224])(crop_tensor)

    
        return crop_tensor,ref_image_tensor,result_img_name



    def __len__(self):
        return self.length



