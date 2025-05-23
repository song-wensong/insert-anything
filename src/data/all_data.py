import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset


class AllDataset(BaseDataset):
    def __init__(self, image_dir, data_type="noperson"):
        super().__init__()
        self.image_root = image_dir 
        self.size = (768,768)
        self.data_type = data_type


    def __len__(self):
        return len(os.listdir(os.path.join(self.image_root, "ref_mask")))

            
    def get_sample(self, idx):

        ref_masks_path = os.path.join(self.image_root, "ref_mask")
        data = os.listdir(ref_masks_path)

        ref_mask_path = os.path.join(self.image_root, "ref_mask", data[idx])
        tar_image_path = ref_mask_path.replace('/ref_mask/', '/tar_image/')
        ref_image_path = ref_mask_path.replace('/ref_mask/','/ref_image/')
        tar_mask_path = ref_mask_path.replace('/ref_mask/', '/tar_mask/')

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = (cv2.imread(tar_mask_path) > 128).astype(np.uint8)[:,:,0]


        item = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask)
        return item

