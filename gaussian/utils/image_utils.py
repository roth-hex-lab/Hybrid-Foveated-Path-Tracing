#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# ---------- basic utilities ----------
# Rec.709 / sRGB luminance coefficients (CPU tensor; moved to device when used)
_LUM_COEFF = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)

# sRGB -> XYZ matrix (D65)
_SRGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=torch.float32)


# -------------------------
# Basic utilities
# -------------------------
def split_rgb_alpha(img):
    """
    Accepts img of shape (3,H,W) or (4,H,W) or (B,3,H,W) or (B,4,H,W).
    Returns (rgb, alpha) with same number of dims as input; alpha is None if not present.
    """
    if img.ndim == 3:
        C = img.shape[0]
        if C == 4:
            return img[:3], img[3:4]
        elif C == 3:
            return img, None
        else:
            raise ValueError(f"split_rgb_alpha: unexpected channel count {C}")
    elif img.ndim == 4:
        C = img.shape[1]
        if C == 4:
            return img[:, :3], img[:, 3:4]
        elif C == 3:
            return img, None
        else:
            raise ValueError(f"split_rgb_alpha: unexpected channel count {C}")
    else:
        raise ValueError(f"split_rgb_alpha: expected 3D or 4D tensor, got ndim={img.ndim}")


def composite_over_black(rgb, alpha):
    """
    Composite rgb over black using alpha.
    rgb: (3,H,W) or (B,3,H,W)
    alpha: None or (1,H,W) or (B,1,H,W)
    Returns same shape as rgb.
    """
    if alpha is None:
        return rgb
    # rely on broadcasting: channel dim of rgb is 3, alpha channel dim is 1
    return rgb * alpha


# -------------------------
# Lightness / L* helpers
# -------------------------
def linearize_pow(rgb, gamma=2.2):
    """Inverse gamma power-law (cheap). Expects rgb in [0,1]."""
    return rgb.clamp(min=0.0, max=1.0).pow(gamma)


def rgb_linear_to_xyz(rgb_lin):
    """
    rgb_lin: (B,3,H,W)
    returns xyz: (B,3,H,W)
    """
    B, C, H, W = rgb_lin.shape
    assert C == 3
    M = _SRGB_TO_XYZ.to(rgb_lin.device).to(rgb_lin.dtype)  # (3,3)
    flat = rgb_lin.view(B, 3, -1)                          # (B,3,HW)
    xyz_flat = torch.matmul(M, flat)                       # (B,3,HW)
    xyz = xyz_flat.view(B, 3, H, W)
    return xyz


def xyz_to_Lstar(xyz, Yn=1.0):
    """
    xyz: (B,3,H,W)
    Returns L*: (B,1,H,W)
    """
    Y = xyz[:, 1:2, :, :]  # (B,1,H,W)
    t = (Y / Yn).clamp(min=0.0)
    delta = 6.0 / 29.0
    t0 = delta ** 3
    f = torch.where(t > t0, t.pow(1.0 / 3.0), (t / (3.0 * delta * delta) + 4.0 / 29.0))
    Lstar = 116.0 * f - 16.0
    return Lstar


def compute_lightness_map(rgb, use_lab=False, gamma_for_linearize=2.2, use_srgb=False):
    """
    Compute a "lightness" map from rgb.
    - rgb: (3,H,W) or (B,3,H,W). Input is assumed **display-encoded** (tonemapped + gamma) in
      the default cheap mode (use_lab=False). If use_lab=True, we linearize and compute L*.
    - use_lab False: returns Rec709-like luminance in same batched shape -> (B,1,H,W) or (1,H,W)
    - use_lab True: returns CIE L* (B,1,H,W) or (1,H,W) after linearization
    """
    # normalize input to batched format
    single = False
    if rgb.ndim == 3:
        rgb_b = rgb.unsqueeze(0)  # (1,3,H,W)
        single = True
    else:
        rgb_b = rgb  # (B,3,H,W)

    if not use_lab:
        coeff = _LUM_COEFF.to(rgb_b.device).to(rgb_b.dtype).view(1, 3, 1, 1)
        lum = (rgb_b * coeff).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return lum.squeeze(0) if single else lum

    # use_lab == True -> linearize then convert to L*
    if use_srgb:
        # sRGB inverse companding
        a = 0.055
        low_mask = (rgb_b <= 0.04045).to(rgb_b.dtype)
        high_mask = 1.0 - low_mask
        rgb_lin = low_mask * (rgb_b / 12.92) + high_mask * (((rgb_b + a) / (1.0 + a)) ** 2.4)
    else:
        rgb_lin = linearize_pow(rgb_b, gamma=gamma_for_linearize)

    xyz = rgb_linear_to_xyz(rgb_lin)  # (B,3,H,W)
    Lstar = xyz_to_Lstar(xyz)        # (B,1,H,W)
    return Lstar.squeeze(0) if single else Lstar


# -------------------------
# Mask helper (either image has content)
# -------------------------
def compute_either_mask(img1, img2, black_threshold=0.0):
    """
    img1, img2: (3,H,W) or (B,3,H,W)
    returns mask of shape (1,H,W) if inputs were 3D, else (B,1,H,W).
    Pixel is valid if any channel > black_threshold in either image.
    """
    single = False
    if img1.ndim == 3:
        a = img1.unsqueeze(0)
        b = img2.unsqueeze(0)
        single = True
    else:
        a = img1
        b = img2

    m1 = (a > black_threshold).any(dim=1, keepdim=True)  # (B,1,H,W)
    m2 = (b > black_threshold).any(dim=1, keepdim=True)
    mask = (m1 | m2).to(dtype=a.dtype, device=a.device)
    return mask.squeeze(0) if single else mask


# -------------------------
# Lightness loss (log-L1) and patch contrast loss
# -------------------------
def lightness_loss(pred_rgb_comp, gt_rgb_comp, mask=None, use_lab=False, gamma=2.2, use_srgb=False, eps=1e-4):
    """
    pred_rgb_comp, gt_rgb_comp: (3,H,W) or (B,3,H,W), display encoded = tonemapped+gamma
    mask: optional (1,H,W) or (B,1,H,W) float (1 valid, 0 ignore).
    Returns scalar loss (tensor).
    """
    # ensure batched internally
    single = False
    if pred_rgb_comp.ndim == 3:
        pred_b = pred_rgb_comp.unsqueeze(0)
        gt_b = gt_rgb_comp.unsqueeze(0)
        single = True
    else:
        pred_b = pred_rgb_comp
        gt_b = gt_rgb_comp

    Lp = compute_lightness_map(pred_b, use_lab=use_lab, gamma_for_linearize=gamma, use_srgb=use_srgb)  # (B,1,H,W)
    Lg = compute_lightness_map(gt_b,   use_lab=use_lab, gamma_for_linearize=gamma, use_srgb=use_srgb)

    # log L1
    lp = torch.log(Lp.clamp(min=eps) + eps)
    lg = torch.log(Lg.clamp(min=eps) + eps)
    diff = (lp - lg).abs()  # (B,1,H,W)

    if mask is not None:
        # make mask batched if needed
        if mask.ndim == 3:
            mask_b = mask.unsqueeze(0)
        else:
            mask_b = mask
        denom = mask_b.sum().clamp(min=1.0)
        loss = (diff * mask_b).sum() / denom
    else:
        loss = diff.mean()

    return loss.squeeze()


def _box_filter(x, k):
    pad = k // 2
    w = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
    # reflect pad to avoid boundary artifacts
    return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'), w, padding=0)


def local_rms_map(Lmap, k=9):
    """
    Lmap: (B,1,H,W)
    returns RMS std map (B,1,H,W)
    """
    mean = _box_filter(Lmap, k)
    mean2 = _box_filter(Lmap * Lmap, k)
    var = (mean2 - mean * mean).clamp(min=0.0)
    return torch.sqrt(var + 1e-8)


def patch_contrast_loss(pred_rgb_comp, gt_rgb_comp, mask=None, k=9, use_lab=False, gamma=2.2, use_srgb=False, edge_weight=True):
    """
    pred_rgb_comp, gt_rgb_comp: (3,H,W) or (B,3,H,W)
    mask: optional (1,H,W) or (B,1,H,W)
    edge_weight: bool; if True use GT edges to prioritize edges.
    Returns scalar loss.
    """
    single = False
    if pred_rgb_comp.ndim == 3:
        pred_b = pred_rgb_comp.unsqueeze(0)
        gt_b = gt_rgb_comp.unsqueeze(0)
        single = True
    else:
        pred_b = pred_rgb_comp
        gt_b = gt_rgb_comp

    Lp = compute_lightness_map(pred_b, use_lab=use_lab, gamma_for_linearize=gamma, use_srgb=use_srgb)  # (B,1,H,W)
    Lg = compute_lightness_map(gt_b,   use_lab=use_lab, gamma_for_linearize=gamma, use_srgb=use_srgb)

    std_p = local_rms_map(Lp, k=k)
    std_g = local_rms_map(Lg, k=k)
    diff = (std_p - std_g).abs()  # (B,1,H,W)

    if edge_weight:
        # sobel on Lg
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=diff.device, dtype=diff.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=diff.device, dtype=diff.dtype).view(1, 1, 3, 3)
        pad = 1
        gx = F.conv2d(F.pad(Lg, (pad, pad, pad, pad), mode='reflect'), sobel_x)
        gy = F.conv2d(F.pad(Lg, (pad, pad, pad, pad), mode='reflect'), sobel_y)
        edge = torch.sqrt(gx * gx + gy * gy)  # (B,1,H,W)
        # normalize edges per image
        edge_flat = edge.view(edge.shape[0], -1)
        max_per_image = edge_flat.max(dim=1)[0].clamp(min=1e-6).view(-1, 1, 1, 1)
        edge_norm = edge / max_per_image
        weight = 0.2 + 0.8 * edge_norm  # keep floor so non-edge regions still contribute a little
        diff = diff * weight

    if mask is not None:
        if mask.ndim == 3:
            mask_b = mask.unsqueeze(0)
        else:
            mask_b = mask
        denom = mask_b.sum().clamp(min=1.0)
        loss = (diff * mask_b).sum() / denom
    else:
        loss = diff.mean()

    return loss.squeeze()