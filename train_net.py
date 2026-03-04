# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
import register_cells

# MaskFormer
from mask2former import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)

from detectron2.engine import hooks
from detectron2.utils import comm
import torch
import cv2
import numpy as np
np.bool = bool
np.int = int
np.float = float
np.complex = complex
np.object = object
np.str = str
'''
class LossTriggeredEvalBest(hooks.HookBase):
    def __init__(
        self,
        checkpointer,
        eval_fn,
        loss_key="total_loss",
        metric_path="sem_seg/mIoU",
        file_prefix="model_best",
        cooldown_iters=200,
        start_iter=500,
    ):
        self.checkpointer = checkpointer
        self.eval_fn = eval_fn
        self.loss_key = loss_key
        self.metric_path = metric_path
        self.file_prefix = file_prefix
        self.cooldown_iters = cooldown_iters
        self.start_iter = start_iter

        self.best_loss = None
        self.best_metric = None
        self._cooldown = 0

    def _get_latest_loss(self):
        try:
            v = self.trainer.storage.history(self.loss_key).latest()
        except Exception:
            return None
        if torch.is_tensor(v):
            v = v.item()
        return float(v)

    def _extract_metric(self, results):
        m = results
        for k in self.metric_path.split("/"):
            m = m[k]
        return float(m)

    def after_step(self):
        if not comm.is_main_process():
            return

        it = self.trainer.iter + 1
        if it < self.start_iter:
            return

        cur_loss = self._get_latest_loss()
        if cur_loss is None:
            return

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        # Trigger eval only when loss is a new minimum
        if self.best_loss is None or cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self._cooldown = self.cooldown_iters

            results = self.eval_fn()
            miou = self._extract_metric(results)

            if self.best_metric is None or miou > self.best_metric:
                self.best_metric = miou
                self.checkpointer.save(self.file_prefix)
                print(f"[BEST] iter={it} mIoU={miou:.4f} loss={cur_loss:.4f} -> saved {self.file_prefix}.pth")
            else:
                print(f"[EVAL] iter={it} mIoU={miou:.4f} (best={self.best_metric:.4f}) loss={cur_loss:.4f}")

'''
class LossTriggeredEvalBest(hooks.HookBase):
    def __init__(
        self,
        checkpointer,
        eval_fn,
        loss_key="total_loss",
        metric_path="sem_seg/mIoU",
        file_prefix="model_best",
        cooldown_iters=50,
        start_iter=500,
    ):
        self.checkpointer = checkpointer
        self.eval_fn = eval_fn
        self.loss_key = loss_key
        self.metric_path = metric_path
        self.file_prefix = file_prefix
        self.cooldown_iters = cooldown_iters
        self.start_iter = start_iter

        self.best_loss = None
        self.best_metric = None
        self._cooldown = 0

    def _get_latest_loss(self):
        try:
            v = self.trainer.storage.history(self.loss_key).latest()
        except Exception:
            return None
        if torch.is_tensor(v):
            v = v.item()
        return float(v)

    def _extract_metric(self, results, path):
        m = results
        for k in path.split("/"):
            if k not in m:
                return None
            m = m[k]
        return float(m)

    def after_step(self):
        if not comm.is_main_process():
            return

        it = self.trainer.iter + 1
        if it < self.start_iter:
            return

        cur_loss = self._get_latest_loss()
        if cur_loss is None:
            return

        if self._cooldown > 0:
            self._cooldown -= 1
            return

        if self.best_loss is None or cur_loss < self.best_loss:
            self.best_loss = cur_loss
            self._cooldown = self.cooldown_iters

            results = self.eval_fn()

            # extract primary metric for saving best model
            primary = self._extract_metric(results, self.metric_path)

            # extract all available metrics for logging
            metrics = {}
            for path in ["segm/AP", "segm/AP50", "segm/AP75", "sem_seg/mIoU", "dice/Dice"]:
                v = self._extract_metric(results, path)
                if v is not None:
                    metrics[path] = v

            metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])

            if primary is not None and (self.best_metric is None or primary > self.best_metric):
                self.best_metric = primary
                self.checkpointer.save(self.file_prefix)
                print(f"[BEST] iter={it} {metrics_str} loss={cur_loss:.4f} -> saved {self.file_prefix}.pth")
            else:
                best_str = f"best_{self.metric_path}={self.best_metric:.4f}" if self.best_metric else ""
                print(f"[EVAL] iter={it} {metrics_str} ({best_str}) loss={cur_loss:.4f}")


class CellsInstanceDatasetMapper(MaskFormerInstanceDatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        if not is_train:
            return

        if cfg.INPUT.CELLS_AUG:
            extra = [
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
            if hasattr(self, "tfm_gens") and self.tfm_gens is not None:
                self.tfm_gens = extra + list(self.tfm_gens)
            elif hasattr(self, "augmentations") and self.augmentations is not None:
                self.augmentations = extra + list(self.augmentations)
            else:
                raise AttributeError(
                    "MaskFormerInstanceDatasetMapper has neither `tfm_gens` nor `augmentations`.")
        else:
            if hasattr(self, "tfm_gens"):
                self.tfm_gens = [t for t in self.tfm_gens if not isinstance(t, T.RandomFlip)]
            elif hasattr(self, "augmentations"):
                self.augmentations = [t for t in self.augmentations if not isinstance(t, T.RandomFlip)]


class CellsSemanticDatasetMapper(MaskFormerSemanticDatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        if not is_train:
            return

        if cfg.INPUT.CELLS_AUG:
            extra = [
                T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ]
            if hasattr(self, "tfm_gens") and self.tfm_gens is not None:
                self.tfm_gens = extra + list(self.tfm_gens)
            elif hasattr(self, "augmentations") and self.augmentations is not None:
                self.augmentations = extra + list(self.augmentations)
            else:
                raise AttributeError(
                    "MaskFormerSemanticDatasetMapper has neither `tfm_gens` nor `augmentations`.")
        else:
            if hasattr(self, "tfm_gens"):
                self.tfm_gens = [t for t in self.tfm_gens if not isinstance(t, T.RandomFlip)]
            elif hasattr(self, "augmentations"):
                self.augmentations = [t for t in self.augmentations if not isinstance(t, T.RandomFlip)]
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
            evaluator_list.append(DiceEvaluator())  # only for semantic

        # instance segmentation
        '''
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        '''
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))

        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type))
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):


        if cfg.INPUT.DATASET_MAPPER_NAME == "cells_semantic":
            mapper = CellsSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        if cfg.INPUT.DATASET_MAPPER_NAME == "cells_instance":
            mapper = CellsInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)   

        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})
        def build_writers(self):
            """
            Disable TensorboardXWriter to avoid torch.utils.tensorboard importing distutils.
            """
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            ]
        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    '''
    def build_hooks(self):
        hooks_list = super().build_hooks()

        # You disabled default eval by TEST.EVAL_PERIOD=0, so only this hook evaluates.
        hooks_list.append(
            LossTriggeredEvalBest(
                checkpointer=self.checkpointer,
                eval_fn=lambda: self.test(self.cfg, self.model),
                loss_key="total_loss",
                metric_path="sem_seg/mIoU",
                file_prefix="model_best",
                cooldown_iters=1,
                start_iter=500,
            )
        )
        return hooks_list

        
    '''
    def build_hooks(self):
        hooks_list = super().build_hooks()
    
        if self.cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            metric_path = "segm/AP"
        else:
            metric_path = "sem_seg/mIoU"
    
        hooks_list.append(
            LossTriggeredEvalBest(
                checkpointer=self.checkpointer,
                eval_fn=lambda: self.test(self.cfg, self.model),
                loss_key="total_loss",
                metric_path=metric_path,
                file_prefix="model_best",
                cooldown_iters=50,
                start_iter=500,
            )
        )
        return hooks_list

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)

    if not hasattr(cfg, "INPUT"):
        cfg.INPUT = CN()
    cfg.INPUT.CELLS_AUG = True  # default: keep online aug ON

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg

from detectron2.evaluation import DatasetEvaluator
from pycocotools import mask as mask_util

from detectron2.evaluation import DatasetEvaluator

class DiceEvaluator(DatasetEvaluator):
    def __init__(self):
        self.intersections = []
        self.unions = []

    def reset(self):
        self.intersections = []
        self.unions = []

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            if "sem_seg" not in out:
                continue
            pred = (out["sem_seg"].argmax(0).cpu().numpy() > 0).astype(np.uint8)
            gt = inp["sem_seg"].numpy()

            # resize pred to match GT if needed
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # mask out ignore regions
            valid = (gt != 255)
            pred = pred[valid]
            gt = (gt[valid] > 0).astype(np.uint8)

            intersection = (pred & gt).sum()
            union = pred.sum() + gt.sum()
            self.intersections.append(intersection)
            self.unions.append(union)

    def evaluate(self):
        total_inter = sum(self.intersections)
        total_union = sum(self.unions)
        dice = 2.0 * total_inter / (total_union + 1e-6)
        return {"dice": {"Dice": float(dice)}}

def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
            
            # Print clean summary
            print("\n" + "="*40)
            print("EVALUATION SUMMARY")
            print("="*40)
            if "sem_seg" in res:
                print(f"  mIoU:      {res['sem_seg']['mIoU']:.4f}")
                print(f"  Cell IoU:  {res['sem_seg']['IoU-cell']:.4f}")
            if "dice" in res:
                print(f"  Dice:      {res['dice']['Dice']:.4f}")
            if "segm" in res:
                print(f"  AP:        {res['segm']['AP']:.4f}")
                print(f"  AP50:      {res['segm']['AP50']:.4f}")
                print(f"  AP75:      {res['segm']['AP75']:.4f}")
            print("="*40)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
