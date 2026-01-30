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

import math
from pathlib import Path
import os
import re
import statistics
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, create_window
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import flip_evaluator as flip


def compute_mask(img1, img2, black_threshold=0.0):
    """
    Computes a mask for pixels that have information in either of the two images.
    A pixel is considered to have information if it is not black.
    """
    img1_has_info = torch.any(img1 > black_threshold, dim=1, keepdim=True)
    img2_has_info = torch.any(img2 > black_threshold, dim=1, keepdim=True)
    mask = img1_has_info | img2_has_info
    return mask


def masked_psnr(img1, img2, mask):
    """
    Computes PSNR on the pixels specified by the mask.
    """
    num_pixels_per_image = torch.sum(mask, dim=[2, 3], keepdim=True)
    squared_error = (img1 - img2) ** 2
    masked_squared_error = squared_error * mask.float()
    sum_error_per_image = torch.sum(masked_squared_error, dim=[1, 2, 3], keepdim=True)
    num_vals_per_image = num_pixels_per_image * img1.shape[1]
    masked_mse = sum_error_per_image / (num_vals_per_image + 1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(masked_mse))


def masked_ssim(img1, img2, mask, window_size=11):
    """
    Computes SSIM on the pixels specified by the mask.
    It calculates the full SSIM map and then averages only the unmasked pixel values.
    """
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    num_pixels_per_image = torch.sum(mask, dim=[2, 3], keepdim=True)
    masked_ssim_map = ssim_map * mask.float()
    ssim_sum = torch.sum(masked_ssim_map, dim=[1, 2, 3], keepdim=True)
    num_vals_per_image = num_pixels_per_image * img1.shape[1]
    mean_ssim = ssim_sum / (num_vals_per_image + 1e-10)
    return mean_ssim.squeeze(-1).squeeze(-1)


def flip_metric(render_tensor, gt_tensor, ppd = None):
    """
    Computes FLIP metric between two torch tensors.
    Returns the mean FLIP error, or None if computation fails.
    
    Args:
        render_tensor: torch.Tensor of shape [1, C, H, W] on CUDA
        gt_tensor: torch.Tensor of shape [1, C, H, W] on CUDA
    """

    # Move to CPU and remove batch dimension [1, C, H, W] -> [C, H, W]
    render_np = render_tensor.squeeze(0).detach().cpu().numpy()
    gt_np = gt_tensor.squeeze(0).detach().cpu().numpy()

    # Transpose from CHW to HWC
    render_np = render_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    gt_np = gt_np.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
    
    image_type = "LDR"
    parameters = {}
    if ppd:
        parameters["ppd"] = ppd    

    # FLIP evaluate with numpy arrays - note the order: reference first, test second
    _, mean_flip_error, _ = flip.evaluate(gt_np, render_np, image_type, parameters=parameters)
    return mean_flip_error



def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        renders.append(renders_dir / fname)   # paths; load on demand
        gts.append(gt_dir / fname)
        image_names.append(fname)
    return renders, gts, image_names


def read_ply_gs_count(path: str) -> int:
    with open(path, "rb") as f:  # open in binary so it works for both ascii and binary PLY
        first_line = f.readline().decode("ascii").strip()
        if first_line != "ply":
            raise ValueError(f"Not a PLY file (first line: {first_line})")

        vertex_count = 0
        for line in f:
            line = line.decode("ascii").strip()
            if line.startswith("element vertex"):
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Malformed element line: {line}")
                vertex_count = int(parts[2])
                break
            elif line == "end_header":
                break

        if not vertex_count:
            raise ValueError(f"Could not find vertex count in {path}")

        return vertex_count


def evaluate(model_paths, subpic):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
    
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}
        full_dict_polytopeonly[scene_dir] = {}
        per_view_dict_polytopeonly[scene_dir] = {}

        test_dir = Path(scene_dir) / "test"

        for method_name in os.listdir(test_dir):

            modes = [("full", method_name)]
            if isinstance(subpic, int):
                modes.append(("sub", f"{method_name}_sub_{subpic}"))

            for mode, method in modes:

                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method_name
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                masked_ssims = []
                masked_psnrs = []

                flip_25 = []
                flip_50 = []

                try:
                    matched_iter = re.search(r"ours_(\d+)", method)
                    iteration = int(matched_iter.group(1))
                    pc_path = os.path.join(scene_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
                    gs_count = read_ply_gs_count(pc_path)
                except Exception as e:
                    print(f"  Warning: Could not read GS count from PLY: {e}")
                    gs_count = 0

                skiptcount = 0

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render_img = Image.open(renders[idx]).convert("RGB")
                    gt_img = Image.open(gts[idx]).convert("RGB")

                    if mode == 'sub':
                        width, height = render_img.size
                        left = (width - subpic) / 2
                        top = (height - subpic) / 2
                        right = (width + subpic) / 2
                        bottom = (height + subpic) / 2
                        render_img = render_img.crop((left, top, right, bottom))
                        gt_img = gt_img.crop((left, top, right, bottom))

                        render_img.save(method_dir / f"render_sub_{subpic}_{image_names[idx]}.png")
                        gt_img.save(method_dir / f"gt_sub_{subpic}_{image_names[idx]}.png")

                    render = tf.to_tensor(render_img).unsqueeze(0).cuda()
                    gt = tf.to_tensor(gt_img).unsqueeze(0).cuda()

                    psnr_score = psnr(render, gt)
                    if not math.isfinite(psnr_score.item()) or psnr_score.item() > 75:
                        skiptcount += 1
                        continue

                    ssims.append(ssim(render, gt))
                    psnrs.append(psnr_score)
                    lpipss.append(lpips(render, gt, net_type='vgg'))

                    mask = compute_mask(render, gt, black_threshold=0.0)
                    masked_ssims.append(masked_ssim(render, gt, mask))
                    masked_psnrs.append(masked_psnr(render, gt, mask))

                    mean_flip_25 = flip_metric(render, gt, ppd = 25)
                    mean_flip_50 = flip_metric(render, gt, ppd = 50)
                    flip_25.append(mean_flip_25)
                    flip_50.append(mean_flip_50)

                if skiptcount:
                    print(f"Skipped {skiptcount} empty images")

                ssim_stdev = psnr_stdev = lpips_stdev = 0
                ssim_m_stdev = psnr_m_stdev = 0
                flip25_stdev = flip50_stdev = 0

                if len(ssims) > 1:
                    ssim_stdev = statistics.stdev([ele.item() for ele in ssims])
                    psnr_stdev = statistics.stdev([ele.item() for ele in psnrs])
                    lpips_stdev = statistics.stdev([ele.item() for ele in lpipss])
                    ssim_m_stdev = statistics.stdev([ele.item() for ele in masked_ssims])
                    psnr_m_stdev = statistics.stdev([ele.item() for ele in masked_psnrs])
                    flip25_stdev = statistics.stdev(flip_25)
                    flip50_stdev = statistics.stdev(flip_50)

                print(f"  SSIM        : {torch.tensor(ssims).mean():>10.5f}±{ssim_stdev:.5f}")
                print(f"  PSNR        : {torch.tensor(psnrs).mean():>10.5f}±{psnr_stdev:.5f}")
                print(f"  LPIPS       : {torch.tensor(lpipss).mean():>10.5f}±{lpips_stdev:.5f}")
                print(f"  Masked SSIM : {torch.tensor(masked_ssims).mean():>10.5f}±{ssim_m_stdev:.5f}")
                print(f"  Masked PSNR : {torch.tensor(masked_psnrs).mean():>10.5f}±{psnr_m_stdev:.5f}")
                print(f"  FLIP_25PPD  : {sum(flip_25)/len(flip_25):>10.5f}±{flip25_stdev:.5f}")
                print(f"  FLIP_50PPD  : {sum(flip_50)/len(flip_50):>10.5f}±{flip50_stdev:.5f}")
                print(f"  GS COUNT    : {gs_count}")
                print("")

                # Aggregate for JSON
                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "Masked_SSIM": torch.tensor(masked_ssims).mean().item(),
                    "Masked_PSNR": torch.tensor(masked_psnrs).mean().item(),
                    "SSIM_STDEV": ssim_stdev,
                    "PSNR_STDEV": psnr_stdev,
                    "LPIPS_STDEV": lpips_stdev,
                    "Masked_SSIM_STDEV": ssim_m_stdev,
                    "Masked_PSNR_STDEV": psnr_m_stdev,
                    "FLIP_25PPD": (sum(flip_25)/len(flip_25)) if flip_25 else None,
                    "FLIP_25PPD_STDEV": flip25_stdev if flip_25 else None,
                    "FLIP_50PPD": (sum(flip_50)/len(flip_50)) if flip_50 else None,
                    "FLIP_50PPD_STDEV": flip50_stdev if flip_50 else None,
                    "GS_COUNT": gs_count
                })

                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "Masked_SSIM": {name: mssim for mssim, name in zip(torch.tensor(masked_ssims).tolist(), image_names)},
                    "Masked_PSNR": {name: mpsnr for mpsnr, name in zip(torch.tensor(masked_psnrs).tolist(), image_names)},
                    "FLIP_25PPD": {name: val for val, name in zip(flip_25, image_names[:len(flip_25)])},
                    "FLIP_50PPD": {name: val for val, name in zip(flip_50, image_names[:len(flip_50)])},
                    "GS_COUNT": gs_count
                })

        with open(scene_dir + "/results.json", 'w') as fp:
            json.dump(full_dict[scene_dir], fp, indent=True)
        with open(scene_dir + "/per_view.json", 'w') as fp:
            json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--subpic', '-s', required=False, type=int)
    args = parser.parse_args()
    evaluate(args.model_paths, args.subpic)
