import os
import pandas as pd
import cv2
import numpy as np
import json

parquet_dir = "path/to/mask-prompt"  
output_dir = "data"
category_label_maps = {}  

print("Processing all parquet files...")

for filename in os.listdir(parquet_dir):
    if filename.endswith(".parquet"):
        parquet_path = os.path.join(parquet_dir, filename)
        df = pd.read_parquet(parquet_path)

        for i, row in df.iterrows():
            split = str(row.get("split", "unknown")) 
            category = str(row.get("category", "unknown"))
            label = str(row.get("label", "unknown"))
            sample_id_full = str(row.get("id", f"{i:06}"))
            sample_id = sample_id_full.split("/")[-1]
            image_name = f"{sample_id}.png"

            key = (split, category)
            if key not in category_label_maps:
                category_label_maps[key] = {}

            for field in ["ref_image", "ref_mask", "tar_image", "tar_mask"]:
                img_dict = row[field]
                image_bytes = img_dict["bytes"]
                image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

                save_dir = os.path.join(output_dir, split, category, field)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, image_name)
                cv2.imwrite(save_path, image_np)

            category_label_maps[key][image_name] = label


for (split, category), label_dict in category_label_maps.items():
    save_dir = os.path.join(output_dir, split, category)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "labels.json")
    with open(save_path, "w") as f:
        json.dump(label_dict, f, indent=2)

print("✅ all images and labels have been successfully saved in the corresponding category folders!")
