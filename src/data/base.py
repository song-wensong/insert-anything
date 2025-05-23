import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import cv2
from .data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torchvision import transforms, utils
import bezier
import random
import torchvision.transforms as T

class BaseDataset(Dataset):
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __len__(self):
        # We adjust the ratio of different dataset by setting the length.
        pass


    def __getitem__(self, idx):
        item = self.get_sample(idx)
        return item
 

    def get_sample(self, idx):
        # Implemented for each specific dataset
        pass


    def process_pairs(self, ref_image, ref_mask, tar_image, tar_mask):
        
        # ========= ref ===========
        # Get the outline Box of the reference image
        ref_box_yyxx = get_bbox_from_mask(ref_mask)
  
        # Filtering background for the reference image
        ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
        masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3) 

        y1,y2,x1,x2 = ref_box_yyxx
        masked_ref_image = masked_ref_image[y1:y2,x1:x2,:] 
        ref_mask = ref_mask[y1:y2,x1:x2] 

        ratio = np.random.randint(11, 15) / 10 
        masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)

        # Padding reference image to square and resize to self.size
        masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False) 
        masked_ref_image = cv2.resize(masked_ref_image.astype(np.uint8), self.size).astype(np.uint8)


        # ========= Training Target ===========
        tar_box_yyxx = get_bbox_from_mask(tar_mask)
        expand_tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #1.1  1.3


        y1,y2,x1,x2 = tar_box_yyxx
        e_y1, e_y2, e_x1, e_x2 = expand_tar_box_yyxx


        if self.data_type == "person":
            prob_bezier = 0.4
            prob_box = 0.8
        else:
            prob_bezier = 0.5
            prob_box = 1.0

        prob=random.uniform(0, 1)

        if prob <= prob_bezier:

            tar_mask = Image.new('RGB', (tar_image.shape[1], tar_image.shape[0]), (0, 0, 0)) 
            top_nodes = np.asfortranarray([
                            [x1,(x1+x2)/2 , x2],
                            [y1, e_y1, y1],
                        ])
            down_nodes = np.asfortranarray([
                    [x2,(x1+x2)/2 , x1],
                    [y2, e_y2, y2],
                ])
            left_nodes = np.asfortranarray([
                    [x1,e_x1 , x1],
                    [y2, (y1+y2)/2, y1],
                ])
            right_nodes = np.asfortranarray([
                    [x2,e_x2 , x2],
                    [y1, (y1+y2)/2, y2],
                ])
            top_curve = bezier.Curve(top_nodes,degree=2)
            right_curve = bezier.Curve(right_nodes,degree=2)
            down_curve = bezier.Curve(down_nodes,degree=2)
            left_curve = bezier.Curve(left_nodes,degree=2)
            curve_list=[top_curve,right_curve,down_curve,left_curve]
            pt_list=[]
            random_width=40
            for curve in curve_list:
                x_list=[]
                y_list=[]
                for i in range(1, 19):
            
                    x_original = curve.evaluate(i * 0.05)[0][0]
                    y_original = curve.evaluate(i * 0.05)[1][0]
            
                    random_x_offset = random.randint(-random_width, random_width)
                    random_y_offset = random.randint(-random_width, random_width)
            
                    x = x_original + random_x_offset
                    y = y_original + random_y_offset


                    x_in_range = x < x1 or x > x2
                    y_in_range = y < y1 or y > y2

                    if not x_in_range:
                        x = x_original
                    if not y_in_range:
                        y = y_original

                    if (x, y) not in zip(x_list, y_list):
                        pt_list.append((x, y))
                        x_list.append(x)
                        y_list.append(y)

            tar_mask_draw=ImageDraw.Draw(tar_mask)
            tar_mask_draw.polygon(pt_list,fill=(255,255,255))

            tar_mask = np.array(tar_mask)

        elif prob > prob_bezier and prob <= prob_box:
            tar_mask = np.zeros_like(tar_image, dtype=np.uint8)
            tar_mask[e_y1:e_y2, e_x1:e_x2] = 255

            
        else:

            tar_mask[tar_mask == 1] = 255

            kernel = np.ones((7, 7), np.uint8)
            iterations = 2
            tar_mask = cv2.dilate(tar_mask, kernel, iterations=iterations)
            tar_mask = np.stack([tar_mask,tar_mask,tar_mask],-1)



        masked_task_image = tar_image * (1-(tar_mask == 255))
        masked_task_image = pad_to_square(masked_task_image, pad_value = 255, random = False).astype(np.uint8)
        masked_task_image = cv2.resize(masked_task_image.astype(np.uint8), self.size).astype(np.uint8)


        tar_image = pad_to_square(tar_image, pad_value = 255, random = False).astype(np.uint8)
        tar_image = cv2.resize(tar_image.astype(np.uint8), self.size).astype(np.uint8)


        tar_mask = pad_to_square(tar_mask, pad_value = 0, random = False).astype(np.uint8)
        tar_mask = cv2.resize(tar_mask.astype(np.uint8), self.size).astype(np.uint8)

        mask_black = np.ones_like(tar_image) * 0
        diptych_mask = self.to_tensor(np.concatenate([mask_black, tar_mask], axis=1))
        diptych_src = self.to_tensor(np.concatenate([masked_ref_image, masked_task_image], axis=1))
        diptych_result = self.to_tensor(np.concatenate([masked_ref_image, tar_image], axis=1))
        masked_ref_image = self.to_tensor(masked_ref_image)


        item = dict(
                ref=masked_ref_image,   
                src=diptych_src, 
                result=diptych_result,
                mask=diptych_mask,  
                ) 
        return item





