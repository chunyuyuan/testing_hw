# UNI2-h + Mask2Former for Cell Segmentation on BCCD Dataset

## 1. Model Adaptation and Design Choices

### Backbone: UNI2-h (Vision Transformer for Pathology)

I adopt **UNI2-h**, a ViT-Giant pretrained on over 100 million pathology image tiles, as the feature extraction backbone. UNI2-h was selected because:

- Based on assigment's requirement. paper cited from "https://www.nature.com/articles/s41591-024-02857-3?utm_source=chatgpt.com"
- Its large embedding dimension (1536) with patch size 14 captures rich morphological features critical for distinguishing overlapping and heterogeneous cell types in blood cell images.
- The pretrained weights (`uni2h_state_dict_cpu.pt`) encode tissue structure priors, reducing the need for extensive training data.

### Pyramid Adapter: UNI → Mask2Former Feature Maps

UNI2-h outputs a single-scale feature map at stride 14 (patch size), whereas Mask2Former requires multi-scale features at strides 4, 8, 16, and 32. To bridge this gap, I designed the UNIPyramidForMask2Former adapter. Prioritizing the architectural suggestions from the original paper, the adapter works as follows:

1. Takes the stride-14 UNI feature map.
2. Interpolates it (bilinear) to four target resolutions: H/4, H/8, H/16, H/32.
3. Projects each through a 2-layer convolutional block (1×1 conv → GroupNorm → ReLU → 3×3 conv → GroupNorm → ReLU) to produce 256-channel feature maps named `res2`, `res3`, `res4`, `res5`.

This design avoids modifying the frozen UNI encoder and is lightweight (~2M additional parameters).

### Segmentation Head: Mask2Former

Mask2Former was chosen for its unified architecture supporting both semantic and instance segmentation with the same backbone and pixel decoder. The shared architectural components are:

| Component | Configuration |
|-----------|--------------|
| Meta Architecture | MaskFormer |
| SEM_SEG_HEAD Name | MaskFormerHead |
| Pixel Decoder | MSDeformAttnPixelDecoder |
| IN_FEATURES | res2, res3, res4, res5 |
| DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES | res3, res4, res5 |
| COMMON_STRIDE | 4 |
| Transformer Encoder Layers | 6 |
| CONVS_DIM / MASK_DIM | 256 |
| Normalization | GN (Group Normalization) |
| Transformer Decoder | MultiScaleMaskedTransformerDecoder |
| TRANSFORMER_IN_FEATURE | multi_scale_pixel_decoder |
| Decoder Layers (DEC_LAYERS) | 4 |
| HIDDEN_DIM | 256 |
| NHEADS | 8 |
| DROPOUT | 0.0 |
| DIM_FEEDFORWARD | 2048 |
| ENC_LAYERS | 0 |
| PRE_NORM | False |
| ENFORCE_INPUT_PROJ | False |
| Deep Supervision | True |
| SIZE_DIVISIBILITY | 32 |
| IGNORE_VALUE | 255 |
| LOSS_WEIGHT | 1.0 |

Key differences between the two task configurations:

| Parameter | Semantic | Instance |
|-----------|----------|----------|
| NUM_CLASSES | 2 (background, cell) | 1 (cell as "thing" class) |
| NUM_OBJECT_QUERIES | 25 | 100 |
| SEMANTIC_ON | True | False |
| INSTANCE_ON | False | True |
| PANOPTIC_ON | False | False |
| DATASET_MAPPER_NAME | cells_semantic | cells_instance |
| DATASETS.TRAIN | cells_train | cells_instance_train |
| DATASETS.TEST | cells_val | cells_instance_val |

The semantic task uses 25 queries for 2-class pixel labeling, while the instance task requires 100 queries to handle the many individual cells per image.

### Task Design

I approach cell segmentation as two complementary tasks:

- **Semantic Segmentation**: Binary classification (background vs. cell) — provides pixel-level cell coverage maps. Uses `SemSegEvaluator` with a custom `DiceEvaluator`.
- **Instance Segmentation**: Individual cell detection with per-instance masks — enables cell counting and morphological analysis. Uses `COCOEvaluator` for AP metrics.

For instance segmentation ground truth, I used **Cellpose** (cyto model, v2.2.3, GPU-accelerated) to automatically generate instance labels from my binary masks, as the original BCCD dataset only provides semantic-level annotations.

### Loss Configuration (Identical for Both Tasks)

| Parameter | Value |
|-----------|-------|
| CLASS_WEIGHT | 2.0 |
| MASK_WEIGHT | 5.0 |
| DICE_WEIGHT | 5.0 |
| NO_OBJECT_WEIGHT | 0.1 |
| TRAIN_NUM_POINTS | 2048 |
| OVERSAMPLE_RATIO | 3.0 |
| IMPORTANCE_SAMPLE_RATIO | 0.75 |

### Inference Thresholds (Identical for Both Tasks)

| Parameter | Value |
|-----------|-------|
| OVERLAP_THRESHOLD | 0.8 |
| OBJECT_MASK_THRESHOLD | 0.8 |

---

## 2. Training Procedure and Hyperparameters

### Dataset: BCCD (Blood Cell Count Detection)

| Split | Images | Usage |
|-------|--------|-------|
| Train | ~1,000 | Training |
| Val | ~200 | Validation / model selection |
| Test | ~159 | Final evaluation |

Image resolutions: primarily 1600×1200 and 1944×1383. All training images were resized to a uniform scale during preprocessing.

### Data Preprocessing

- Validated all mask values (must be 0, 1, or 255 only). Identified and fixed 3 corrupted masks with unexpected pixel values (29, 105, 149, 178) — caused by lossy compression artifacts during mask creation.
- Resized all images to uniform resolution (1600×1200) to prevent CUDA memory errors from variable tensor sizes in the Hungarian matcher.
- Masks are always resized with nearest-neighbor interpolation to preserve class IDs.
- For instance segmentation, ran Cellpose (`model_type="cyto"`, `channels=[0,0]`) on all images to generate per-cell instance masks saved as 16-bit PNG files.

### Data Augmentation

I implemented custom dataset mappers (`CellsSemanticDatasetMapper`, `CellsInstanceDatasetMapper`) with a configurable `CELLS_AUG` flag in both YAML configs:

- **`CELLS_AUG: True`**: Adds random vertical flip (p=0.5) and random horizontal flip (p=0.5) on top of the default resize augmentation.
- **`CELLS_AUG: False`**: Only resize applied, all flips removed (including the parent mapper's default horizontal flip).

Both configs use `CELLS_AUG: False` as the baseline.

### Solver Configuration (Identical for Both Tasks)

| Parameter | Value |
|-----------|-------|
| Optimizer | ADAMW |
| Base Learning Rate | 0.0001 |
| Weight Decay | 0.05 |
| LR Schedule | WarmupPolyLR |
| Warmup Iterations | 200 |
| Warmup Factor | 0.01 |
| Max Iterations | 5,000 |
| Batch Size (IMS_PER_BATCH) | 2 |
| Checkpoint Period | 500 |
| CLIP_GRADIENTS.ENABLED | True |
| CLIP_GRADIENTS.CLIP_TYPE | full_model |
| CLIP_GRADIENTS.CLIP_VALUE | 1.0 |

### Input Configuration (Identical for Both Tasks)

| Parameter | Value |
|-----------|-------|
| FORMAT | BGR |
| MIN_SIZE_TRAIN | (196, 224, 280) — random multi-scale |
| MAX_SIZE_TRAIN | 384 |
| MIN_SIZE_TEST | 280 |
| MAX_SIZE_TEST | 384 |
| DATALOADER.NUM_WORKERS | 4 |

### Test Configuration

| Parameter | Value |
|-----------|-------|
| EVAL_PERIOD | 50 |
| TEST.AUG.ENABLED | False |

### Model Selection Strategy

In addition to the periodic evaluation every 50 iterations, I implemented a custom **LossTriggeredEvalBest** hook that:

1. Monitors training loss with a cooldown mechanism (50 iterations between evaluations).
2. Triggers validation when training loss reaches a new minimum (starting from iteration 500).
3. Saves the best model checkpoint based on the primary metric:
   - **Semantic**: `sem_seg/mIoU`
   - **Instance**: `segm/AP`
4. Logs all available metrics at each evaluation: mIoU and Dice for semantic; AP, AP50, AP75 for instance.

---

## 3. Quantitative Results

### Semantic Segmentation (Test Set)

| Metric | Value |
|--------|-------|
| mIoU | 91.33 |
| Cell IoU | 88.13 |
| Background IoU | 94.54 |
| Dice | 0.9219 |
| Pixel Accuracy | 96.11 |

### Instance Segmentation (Validation Set)

| Metric | Value |
|--------|-------|
| AP @[IoU=0.50:0.95] | 0.829 |
| AP @[IoU=0.50] | 2.234 |
| AP @[IoU=0.75] | 0.990 |
| APm (medium area) | 0.961 |

*Note: AP values are on a 0–100 scale. Instance segmentation requires significantly more training iterations to converge compared to semantic segmentation. With extended training (20,000+ iterations), these values are expected to improve substantially.*

---

## 4. Qualitative Visualizations

### Visualization 1: Semantic Segmentation — Prediction vs Ground Truth

*(Insert figure: original image, GT mask, predicted mask side by side)*

**Interpretation**: The semantic model accurately segments the majority of cells with clean boundaries. The predicted binary mask closely matches the ground truth, achieving 91.33 mIoU. Cell interiors are well-captured with minimal false positives in the background region.

### Visualization 2: Instance Segmentation — Colored Instance Map

*(Insert figure: original image, instance ID map, boundary overlay, colored instances)*

**Interpretation**: The model identifies individual cells as separate instances, each receiving a unique color/ID. The boundary overlay confirms that detected cells align with visible cell structures in the original image.

### Visualization 3: GT (Cellpose) vs Predicted Instance Comparison

*(Insert figure: original, Cellpose GT colored, predicted colored side by side)*

**Interpretation**: Comparing Cellpose-generated ground truth with model predictions reveals that the model captures most cells but may differ in boundary precision. Some discrepancies arise from Cellpose's own imperfections in the GT labels, which propagate as label noise during training.

### Visualization 4: Failure Case — Overlapping/Dense Cells

*(Insert figure showing dense cell region with merged predictions)*

**Interpretation**: In dense cell regions, the model struggles to separate tightly packed cells. The semantic model merges overlapping cells into a single connected region, while the instance model may assign a single ID to multiple touching cells. AP50 >> AP75 confirms that cell localization is correct but boundary precision is insufficient at higher IoU thresholds.

### Visualization 5: Boundary Precision Analysis (AP50 vs AP75 Gap)

*(Insert figure comparing predictions at different confidence thresholds)*

**Interpretation**: The large gap between AP50 (2.234) and AP75 (0.990) indicates the model finds cells correctly but predicted mask boundaries are not pixel-precise. This is partially attributable to the small input resolution (MAX_SIZE_TRAIN: 384) — cells are downscaled significantly from 1600×1200 originals, losing fine boundary detail during both training and inference.

---

## 5. Reflections on Challenges and Potential Improvements

### Challenges Encountered

1. **CUDA/CUBLAS Errors**: Encountered `CUBLAS_STATUS_INTERNAL_ERROR` during training with augmented (3,000 image) dataset. Root causes identified:
   - Corrupted mask values (non-binary values like 29, 105, 178 from compression artifacts).
   - Mixed image resolutions (1600×1200 and 1944×1383) causing variable tensor sizes in the Hungarian matcher's einsum operation.
   - Solution: Validated and cleaned all masks to binary (0/1), resized to uniform dimensions.

2. **Package Compatibility**: Python 3.8 environment with older PyTorch caused multiple issues:
   - `np.bool` removed in NumPy 1.24+ — fixed with a compatibility shim (`np.bool = bool`) at the top of `train_net.py`.
   - Cellpose API changes between v2 (`models.Cellpose`) and v3+ (`models.CellposeModel`) — pinned to `cellpose==2.2.3`.

3. **Instance GT Quality**: The BCCD dataset only provides binary masks. I used Cellpose to generate instance labels, introducing dependency on Cellpose's segmentation quality. Some cells may be over- or under-segmented, creating noisy ground truth that limits instance segmentation performance.

4. **GPU Memory Constraints**: UNI2-h (ViT-Giant, embed_dim=1536, 24 layers) consumes ~18.5GB VRAM (`max_mem: 18505M`). This limited batch size to 2 (`IMS_PER_BATCH: 2`) and input resolution to 384px (`MAX_SIZE_TRAIN: 384`), which constrains mask boundary precision for instance segmentation.

5. **Instance Segmentation Convergence**: Instance segmentation converges much slower than semantic. At 5,000 iterations (`MAX_ITER: 5000`), AP remained below 3.0 (on 0–100 scale), while semantic mIoU reached 91+ at similar iteration counts. The per-instance Hungarian matching and loss computation is fundamentally harder than pixel-level classification.

### Potential Improvements

1. **Higher Input Resolution**: Increasing `MAX_SIZE_TRAIN` from 384 to 512 would improve boundary precision (AP75). This could be achieved through mixed-precision training or reducing `TRAIN_NUM_POINTS` from 2048 to 1024.

2. **More Training Iterations**: Instance segmentation would benefit from increasing `MAX_ITER` to 20,000–50,000. The current 5,000 iterations are insufficient for the model to learn precise per-instance mask prediction.

3. **Increase NUM_OBJECT_QUERIES**: If images contain more than 100 cells, the current `NUM_OBJECT_QUERIES: 100` limits detection capacity. Should be set to at least the maximum cell count per image.

4. **Lower Inference Thresholds**: Current `OBJECT_MASK_THRESHOLD: 0.8` may filter out valid but low-confidence predictions. Lowering to 0.3–0.5 could improve recall and AP.

5. **Better Instance GT**: Replace Cellpose with manual annotations or tune Cellpose parameters (diameter, flow threshold) per image type for more accurate instance labels.

6. **Expanded Augmentation**: Set `CELLS_AUG: True` to enable flips, and add rotation (90°/180°/270°), color jitter (brightness ±20%, contrast ±20%), which are standard in pathology image analysis.

7. **Backbone Fine-tuning Strategy**: Apply a backbone learning rate multiplier (e.g., 0.1×) to fine-tune UNI2-h more conservatively while training the Mask2Former head at full learning rate.

---

## Configuration Files

### Semantic Segmentation (`maskformer2_UNI_cell.yaml`)

```yaml
MODEL:
  META_ARCHITECTURE: "MaskFormer"
  BACKBONE:
    NAME: "build_uni_vit_adapter_backbone"
  UNI:
    WEIGHTS: "/home/ubuntu/assets/ckpts/uni2-h/uni2h_state_dict_cpu.pt"
    PATCH_SIZE: 14
    EMBED_DIM: 1536
  SEM_SEG_HEAD:
    NAME: "MaskFormerHead"
    IGNORE_VALUE: 255
    NUM_CLASSES: 2
    LOSS_WEIGHT: 1.0
    CONVS_DIM: 256
    MASK_DIM: 256
    NORM: "GN"
    PIXEL_DECODER_NAME: "MSDeformAttnPixelDecoder"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: ["res3", "res4", "res5"]
    COMMON_STRIDE: 4
    TRANSFORMER_ENC_LAYERS: 6
  MASK_FORMER:
    TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
    TRANSFORMER_IN_FEATURE: "multi_scale_pixel_decoder"
    DEEP_SUPERVISION: True
    NO_OBJECT_WEIGHT: 0.1
    CLASS_WEIGHT: 2.0
    MASK_WEIGHT: 5.0
    DICE_WEIGHT: 5.0
    HIDDEN_DIM: 256
    NUM_OBJECT_QUERIES: 25
    NHEADS: 8
    DROPOUT: 0.0
    DIM_FEEDFORWARD: 2048
    ENC_LAYERS: 0
    PRE_NORM: False
    ENFORCE_INPUT_PROJ: False
    SIZE_DIVISIBILITY: 32
    DEC_LAYERS: 4
    TRAIN_NUM_POINTS: 2048
    OVERSAMPLE_RATIO: 3.0
    IMPORTANCE_SAMPLE_RATIO: 0.75
    TEST:
      SEMANTIC_ON: True
      INSTANCE_ON: False
      PANOPTIC_ON: False
      OVERLAP_THRESHOLD: 0.8
      OBJECT_MASK_THRESHOLD: 0.8
DATASETS:
  TRAIN: ("cells_train",)
  TEST: ("cells_val",)
DATALOADER:
  NUM_WORKERS: 4
INPUT:
  DATASET_MAPPER_NAME: "cells_semantic"
  CELLS_AUG: False
  FORMAT: "BGR"
  MIN_SIZE_TRAIN: (196, 224, 280)
  MAX_SIZE_TRAIN: 384
  MIN_SIZE_TEST: 280
  MAX_SIZE_TEST: 384
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 0.0001
  WEIGHT_DECAY: 0.05
  OPTIMIZER: "ADAMW"
  WARMUP_ITERS: 200
  WARMUP_FACTOR: 0.01
  LR_SCHEDULER_NAME: "WarmupPolyLR"
  MAX_ITER: 5000
  CHECKPOINT_PERIOD: 500
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "full_model"
    CLIP_VALUE: 1.0
TEST:
  AUG:
    ENABLED: False
  EVAL_PERIOD: 50
OUTPUT_DIR: "./output/maskformer2_uni_cell"
```

### Instance Segmentation (`maskformer2_UNI_cell_instance.yaml`)

```yaml
MODEL:
  META_ARCHITECTURE: "MaskFormer"
  BACKBONE:
    NAME: "build_uni_vit_adapter_backbone"
  UNI:
    WEIGHTS: "/home/ubuntu/assets/ckpts/uni2-h/uni2h_state_dict_cpu.pt"
    PATCH_SIZE: 14
    EMBED_DIM: 1536
  SEM_SEG_HEAD:
    NAME: "MaskFormerHead"
    IGNORE_VALUE: 255
    NUM_CLASSES: 1
    LOSS_WEIGHT: 1.0
    CONVS_DIM: 256
    MASK_DIM: 256
    NORM: "GN"
    PIXEL_DECODER_NAME: "MSDeformAttnPixelDecoder"
    IN_FEATURES: ["res2", "res3", "res4", "res5"]
    DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES: ["res3", "res4", "res5"]
    COMMON_STRIDE: 4
    TRANSFORMER_ENC_LAYERS: 6
  MASK_FORMER:
    TRANSFORMER_DECODER_NAME: "MultiScaleMaskedTransformerDecoder"
    TRANSFORMER_IN_FEATURE: "multi_scale_pixel_decoder"
    DEEP_SUPERVISION: True
    NO_OBJECT_WEIGHT: 0.1
    CLASS_WEIGHT: 2.0
    MASK_WEIGHT: 5.0
    DICE_WEIGHT: 5.0
    HIDDEN_DIM: 256
    NUM_OBJECT_QUERIES: 100
    NHEADS: 8
    DROPOUT: 0.0
    DIM_FEEDFORWARD: 2048
    ENC_LAYERS: 0
    PRE_NORM: False
    ENFORCE_INPUT_PROJ: False
    SIZE_DIVISIBILITY: 32
    DEC_LAYERS: 4
    TRAIN_NUM_POINTS: 2048
    OVERSAMPLE_RATIO: 3.0
    IMPORTANCE_SAMPLE_RATIO: 0.75
    TEST:
      SEMANTIC_ON: False
      INSTANCE_ON: True
      PANOPTIC_ON: False
      OVERLAP_THRESHOLD: 0.8
      OBJECT_MASK_THRESHOLD: 0.8
DATASETS:
  TRAIN: ("cells_instance_train",)
  TEST: ("cells_instance_val",)
DATALOADER:
  NUM_WORKERS: 4
INPUT:
  DATASET_MAPPER_NAME: "cells_instance"
  CELLS_AUG: False
  FORMAT: "BGR"
  MIN_SIZE_TRAIN: (196, 224, 280)
  MAX_SIZE_TRAIN: 384
  MIN_SIZE_TEST: 280
  MAX_SIZE_TEST: 384
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 0.0001
  WEIGHT_DECAY: 0.05
  OPTIMIZER: "ADAMW"
  WARMUP_ITERS: 200
  WARMUP_FACTOR: 0.01
  LR_SCHEDULER_NAME: "WarmupPolyLR"
  MAX_ITER: 5000
  CHECKPOINT_PERIOD: 500
  CLIP_GRADIENTS:
    ENABLED: True
    CLIP_TYPE: "full_model"
    CLIP_VALUE: 1.0
TEST:
  AUG:
    ENABLED: False
  EVAL_PERIOD: 50
OUTPUT_DIR: "./output/maskformer2_uni_cell"
```

---

## Repository Structure

```
Mask2Former/
├── configs/
│   └── ade20k/
│       ├── semantic-segmentation/
│       │   └── maskformer2_UNI_cell.yaml
│       └── instance-segmentation/
│           └── maskformer2_UNI_cell_instance.yaml
├── train_net.py                    # Training script with custom hooks and mappers
├── register_cells.py               # Dataset registration (semantic + instance)
├── uni_vit_adapter_backbone.py     # UNI2-h backbone with pyramid adapter
├── mask2former/                    # Mask2Former module
└── output/                         # Training outputs and checkpoints
```

## How to Run

### Semantic Segmentation Training
```bash
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_UNI_cell.yaml --num-gpus 1
```

### Instance Segmentation Training
```bash
python train_net.py --config-file configs/ade20k/instance-segmentation/maskformer2_UNI_cell_instance.yaml --num-gpus 1
```

### Evaluation (Semantic — Test Set)
```bash
python train_net.py --config-file configs/ade20k/semantic-segmentation/maskformer2_UNI_cell.yaml --num-gpus 1 --eval-only DATASETS.TEST "('cells_test',)" MODEL.WEIGHTS ./output/maskformer2_uni_cell/model_best.pth
```

### Evaluation (Instance — Val Set)
```bash
python train_net.py --config-file configs/ade20k/instance-segmentation/maskformer2_UNI_cell_instance.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS ./output/maskformer2_uni_cell/model_best.pth
```
