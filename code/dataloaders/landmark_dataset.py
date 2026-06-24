"""
Landmark Dataset for CeLDA+ Training
"""
import json
import os

import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T


class CL_Landmark_Mix(Dataset):
    """
    Landmark detection dataset
    
    Args:
        base_dir: Path to dataset root
        splits: 'train' or 'test'
        augmentation: Albumentations augmentation pipeline
    
    Dataset structure:
        base_dir/
        ├── train/
        │   ├── train_anno.json
        │   ├── image1.jpg
        │   └── ...
        └── test/
            ├── test_anno.json
            ├── image1.jpg
            └── ...
    
    Annotation format (JSON):
        {
            "image_name": {
                "keypoint": [[x1, y1], [x2, y2], ...],
                "age_group": "adult"  # optional
            },
            ...
        }
    """

    def __init__(self, base_dir=None, splits="train", augmentation=None):
        if base_dir is None:
            raise ValueError("base_dir must not be None")

        self.base_dir = base_dir
        self.splits = splits
        self.augmentation = augmentation
        self.json_file = f"{self.splits}_anno.json"

        self.image_list = self._read_annotation_list()
        self.normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _read_annotation_list(self):
        primary_json = os.path.join(self.base_dir, self.splits, self.json_file)
        fallback_json = os.path.join(self.base_dir, self.splits, f"{self.splits}.json")
        test_fallback_json = os.path.join(self.base_dir, self.splits, "test_anno.json")
        train_fallback_json = os.path.join(self.base_dir, self.splits, "train_anno.json")
        val_fallback_json = os.path.join(self.base_dir, self.splits, "val_anno.json")

        candidates = [primary_json, fallback_json, test_fallback_json, train_fallback_json, val_fallback_json]
        json_path = next((p for p in candidates if os.path.exists(p)), None)
        if json_path is None:
            raise FileNotFoundError(
                "Annotation json not found. Checked: "
                + ", ".join(candidates)
            )

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for file_name, info in data.items():
            if "keypoint" not in info:
                raise KeyError(f"Missing 'keypoint' field in annotation for {file_name}")
            records.append(
                {
                    "image_name": file_name,
                    "landmarks": info["keypoint"],
                    "age_group": info.get("age_group") or "mix",
                }
            )
        return records

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        item = self.image_list[idx]
        file_name = item["image_name"]
        keypoints = item["landmarks"]
        age_group = item["age_group"]

        img_path = os.path.join(self.base_dir, self.splits, file_name + ".jpg")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")

        raw_shape = tuple(image.shape)  # (H, W)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.augmentation is not None:
            augmented = self.augmentation(image=image, keypoints=keypoints)
            image = augmented["image"]
            keypoints = augmented["keypoints"]

        keypoints = np.asarray(keypoints, dtype=np.float32)
        image = self.normalize(image)

        return {
            "image": image,
            "keypoints": keypoints,
            "raw_shape": raw_shape,
            "file_name": file_name,
            "age_group": age_group,
        }
