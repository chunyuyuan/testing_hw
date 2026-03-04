
import os, glob
from pathlib import Path
from detectron2.data import DatasetCatalog, MetadataCatalog

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def register_cell_semseg(name, image_dir, mask_dir):
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    def _load():
        ds = []

        # only pick real images (case-insensitive)
        imgs = []
        for p in sorted(image_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                imgs.append(p)

        if len(imgs) == 0:
            raise FileNotFoundError(f"No images found in {image_dir}")

        for img_path in imgs:
            stem = img_path.stem  # safer than splitext for most cases

            # find any mask with same stem (any extension, any case)
            candidates = sorted(mask_dir.glob(stem + ".*"))
            if len(candidates) == 0:
                # optional: also try recursive search if your masks are nested
                candidates = sorted(mask_dir.rglob(stem + ".*"))

            if len(candidates) == 0:
                raise FileNotFoundError(
                    f"Missing mask for {img_path} (stem={stem}) in {mask_dir}"
                )

            # if multiple matches exist, prefer png first (common for masks)
            candidates_sorted = sorted(
                candidates,
                key=lambda p: (p.suffix.lower() != ".png", str(p).lower())
            )
            mask_path = candidates_sorted[0]

            ds.append({
                "file_name": str(img_path),
                "sem_seg_file_name": str(mask_path),
            })

        return ds

    DatasetCatalog.register(name, _load)
    MetadataCatalog.get(name).set(
        stuff_classes=["background", "cell"],
        ignore_label=255,
        evaluator_type="sem_seg",
    )
register_cell_semseg("cells_train", "/home/ubuntu/dataset/BCCD Dataset with mask/split/train/original", "/home/ubuntu/dataset/BCCD Dataset with mask/split/train/masks_id")
register_cell_semseg("cells_val",   "/home/ubuntu/dataset/BCCD Dataset with mask/split/val/original",   "/home/ubuntu/dataset/BCCD Dataset with mask/split/val/masks_id")
register_cell_semseg("cells_test",   "/home/ubuntu/dataset/BCCD Dataset with mask/test/original",   "/home/ubuntu/dataset/BCCD Dataset with mask/test/masks_id")

register_cell_semseg("cells_train_aug",   "/home/ubuntu/dataset/BCCD Dataset with mask/split/train_aug/original_new",   "/home/ubuntu/dataset/BCCD Dataset with mask/split/train_aug/mask_id_new")




import os
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

root = "/home/ubuntu/dataset/BCCD Dataset with mask"


def get_instance_dicts(img_dir, inst_dir):
    dataset_dicts = []
    
    for img_name in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        inst_path = os.path.join(inst_dir, stem + ".png")
        
        if not os.path.exists(inst_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        inst = cv2.imread(inst_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
        
        record = {
            "file_name": os.path.abspath(img_path),
            "image_id": stem,
            "height": h,
            "width": w,
            "annotations": [],
        }
        
        for cell_id in np.unique(inst):
            if cell_id == 0:
                continue
            
            binary = (inst == cell_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            contour = max(contours, key=cv2.contourArea)
            if contour.shape[0] < 4:
                continue
            
            poly = contour.flatten().tolist()
            x, y, bw, bh = cv2.boundingRect(contour)
            
            record["annotations"].append({
                "bbox": [x, y, x + bw, y + bh],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            })
        
        dataset_dicts.append(record)
    
    return dataset_dicts
DatasetCatalog.register("cells_instance_train_aug", lambda: get_instance_dicts(
    f"{root}/split/train_aug/original_new", f"{root}/split/train_aug/instance_new"))
DatasetCatalog.register("cells_instance_train", lambda: get_instance_dicts(
    f"{root}/split/train/original", f"{root}/split/train/instance"))
DatasetCatalog.register("cells_instance_val", lambda: get_instance_dicts(
    f"{root}/split/val/original", f"{root}/split/val/instance"))
DatasetCatalog.register("cells_instance_test", lambda: get_instance_dicts(
    f"{root}/test/original", f"{root}/test/instance"))
MetadataCatalog.get("cells_instance_train_aug").set(
    thing_classes=["cell"], evaluator_type="coco")
MetadataCatalog.get("cells_instance_train").set(
    thing_classes=["cell"], evaluator_type="coco")
MetadataCatalog.get("cells_instance_val").set(
    thing_classes=["cell"], evaluator_type="coco")
MetadataCatalog.get("cells_instance_test").set(
    thing_classes=["cell"], evaluator_type="coco")
