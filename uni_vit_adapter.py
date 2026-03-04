import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.layers import ShapeSpec

import timm

import torch.nn.functional as _F
import timm.models.vision_transformer as _timm_vit

def _resample_abs_pos_embed_torch19_safe(posemb, new_size, old_size=None, num_prefix_tokens=1,
                                        interpolation="bicubic", antialias=True, verbose=False):
    # This is a minimal safe implementation: same behavior, but no antialias for torch<1.11
    # posemb: (1, num_tokens, dim)
    if new_size == old_size:
        return posemb

    if old_size is None:
        # infer old grid from posemb length (excluding prefix tokens)
        ntok = posemb.shape[1] - num_prefix_tokens
        gs = int(ntok ** 0.5)
        old_size = (gs, gs)

    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb

    dim = posemb_grid.shape[-1]
    posemb_grid = posemb_grid.reshape(1, old_size[0], old_size[1], dim).permute(0, 3, 1, 2)

    try:
        posemb_grid = _F.interpolate(posemb_grid, size=new_size, mode=interpolation, antialias=antialias)
    except TypeError:
        # torch1.9 fallback
        posemb_grid = _F.interpolate(posemb_grid, size=new_size, mode=interpolation)

    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, new_size[0] * new_size[1], dim)
    return torch.cat([posemb_prefix, posemb_grid], dim=1)

# Patch the symbol used inside timm vision_transformer
_timm_vit.resample_abs_pos_embed = _resample_abs_pos_embed_torch19_safe

import torch.nn.functional as F

def _build_uni2h_from_cfg(cfg) -> nn.Module:
    # UNI2-h timm config (from your snippet)
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        # Make this compatible across timm versions
        "mlp_layer": getattr(getattr(timm, "layers", timm), "SwiGLUPacked", None),
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    if timm_kwargs["mlp_layer"] is None:
        raise RuntimeError(
            "Your timm version does not provide SwiGLUPacked. "
            "Install a newer timm inside this env, e.g. `python -m pip install --no-user timm==0.9.12`."
        )

    model = timm.create_model(pretrained=False, **timm_kwargs)

    weights_path = cfg.MODEL.UNI.WEIGHTS
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"MODEL.UNI.WEIGHTS not found: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    # robust unwrap if needed
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


class UNIStride14(Backbone):
    """UNI -> stride-14 feature map: {"uni": (B,1536,H/14,W/14)}"""
    def __init__(self, uni: nn.Module, patch_size: int = 14, embed_dim: int = 1536):
        super().__init__()
        self.uni = uni
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self._out_features = ["uni"]
        self._out_feature_strides = {"uni": patch_size}
        self._out_feature_channels = {"uni": embed_dim}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        ps = self.patch_size
        
        # Pad to multiples of patch size (14) so UNI patch embed always works
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h or pad_w:
            # pad: (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)
            H = H + pad_h
            W = W + pad_w
    
        tok = self.uni.forward_features(x)  # expected (B, N, C)
        h = H // ps
        w = W // ps
        hw = h * w
    
        if tok.dim() != 3:
            raise RuntimeError(f"Expected tokens (B,N,C), got {tok.shape}")
    
        # Force patch tokens only (UNI2-h may include reg/cls tokens)
        if tok.shape[-1] != self.embed_dim:
            raise RuntimeError(f"Expected embed_dim={self.embed_dim}, got {tok.shape}")
    
        n = tok.shape[1]
        if n < hw:
            raise RuntimeError(f"Token count {n} < h*w {hw}. Cannot reshape.")
    
        tok = tok[:, -hw:, :]   # <--- THIS LINE MUST BE HERE, RIGHT BEFORE RESHAPE
    
        feat = tok.transpose(1, 2).contiguous().view(B, self.embed_dim, h, w)
        return {"uni": feat}

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {"uni": ShapeSpec(channels=self.embed_dim, stride=self.patch_size)}


class UNIPyramidForMask2Former(Backbone):
    """
    detectron2 v0.6-compatible pyramid adapter:
    outputs res2/res3/res4/res5 (stride 4/8/16/32) for Mask2Former.
    """
    def __init__(self, bottom_up: UNIStride14, out_channels: int = 256):
        super().__init__()
        self.bottom_up = bottom_up
        in_ch = bottom_up.embed_dim

        def proj():
            return nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )

        self.proj_res2 = proj()
        self.proj_res3 = proj()
        self.proj_res4 = proj()
        self.proj_res5 = proj()

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, _, H, W = x.shape
        uni = self.bottom_up(x)["uni"]

        res2 = F.interpolate(uni, size=(H // 4,  W // 4),  mode="bilinear", align_corners=False)
        res3 = F.interpolate(uni, size=(H // 8,  W // 8),  mode="bilinear", align_corners=False)
        res4 = F.interpolate(uni, size=(H // 16, W // 16), mode="bilinear", align_corners=False)
        res5 = F.interpolate(uni, size=(H // 32, W // 32), mode="bilinear", align_corners=False)

        return {
            "res2": self.proj_res2(res2),
            "res3": self.proj_res3(res3),
            "res4": self.proj_res4(res4),
            "res5": self.proj_res5(res5),
        }

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            "res2": ShapeSpec(channels=self._out_feature_channels["res2"], stride=4),
            "res3": ShapeSpec(channels=self._out_feature_channels["res3"], stride=8),
            "res4": ShapeSpec(channels=self._out_feature_channels["res4"], stride=16),
            "res5": ShapeSpec(channels=self._out_feature_channels["res5"], stride=32),
        }


@BACKBONE_REGISTRY.register()
def build_uni_vit_adapter_backbone(cfg, input_shape):
    uni = _build_uni2h_from_cfg(cfg)
    bottom_up = UNIStride14(uni=uni, patch_size=14, embed_dim=1536)
    return UNIPyramidForMask2Former(bottom_up=bottom_up, out_channels=256)
