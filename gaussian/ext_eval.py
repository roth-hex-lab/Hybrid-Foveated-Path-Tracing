import os
from argparse import ArgumentParser
import shutil
import subprocess
import sys

datasets = ["leg_bone", "leg_muscle", "leg_full", "fullbody_bone", "fullbody_organs", "fullbody_muscle"]
variants = [(12, 8), (16, 16)] # (Pics, Spp)
ext_variants = [(64, 4), (64, 16), (512, 4), (512, 16)]
#ext_variants = [(64, 4), (64, 16), (256, 4), (256, 16), (512, 4), (512, 16)]
pretrain_length = 700
save_iters = [2000, 4000]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_copy_speed", action="store_true")
parser.add_argument("--output_path", default="./eval/ext")
parser.add_argument("--data-root")

args, _ = parser.parse_known_args()

all_scenes = []

def run_call(call: str) -> None:
    try:
        subprocess.run(call, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}): {call}", file=sys.stderr)
        sys.exit(e.returncode)


if not args.skip_copy_speed:
    for dataset in datasets:
        speed_source = os.path.join(args.data_root, dataset, "render_speed.csv")
        speed_dst = os.path.join(args.output_path, dataset)
        os.makedirs(speed_dst, exist_ok=True)
        shutil.copy2(speed_source, speed_dst)

# Create base model first
common_args = " --quiet --eval --test_iterations -1 --config_path config/anatomy_fast "
common_args += f" --test_iterations -1 --checkpoint_iterations {pretrain_length} "

pretrained_models = {}

for dataset in datasets:
    for variant in variants:
        pic, sample = variant
        source = os.path.join(args.data_root, dataset, "train", f"{pic}p_{sample}s_0")
        output = os.path.join(args.output_path, dataset, "pre_train", f"{pic}p_{sample}s_0")

        if not args.skip_training:
            os.makedirs(output, exist_ok=True)
            call = f"python msv2/train.py -s {source} -m {output} {common_args}"
            run_call(call)

        pretrained_models[(dataset, pic, sample)] = output


common_args_ext = " --quiet --eval --test_iterations -1 --config_path config/anatomy_cont_smaller "
common_args_ext += f" --test_iterations -1 --save_iterations {' '.join(map(str, save_iters))} "

for pre in pretrained_models:
    dataset, pre_pic, pre_sample = pre
    for ext in ext_variants:
        ext_pic, ext_sample = ext
        output = os.path.join(args.output_path, dataset, "extended", f"{pre_pic}p_{pre_sample}s_ext_{ext_pic}p_{ext_sample}s")

        if not args.skip_training:
            ext_source = os.path.join(args.data_root, dataset, "ext", f"{ext_pic}p_{ext_sample}s_EXT")
            checkpoint = os.path.join(pretrained_models[pre], f"chkpnt{pretrain_length}.pth")
            # Need to copy original training set and merge with extension images (all non-test)
            orig_source = os.path.join(args.data_root, dataset, "train", f"{pre_pic}p_{pre_sample}s_0")
            source_target = os.path.join(args.output_path, dataset, "extended_train", f"{pre_pic}p_{pre_sample}s_ext_{ext_pic}p_{ext_sample}s")
            shutil.copytree(orig_source, source_target, dirs_exist_ok=True)

            merge_call = f"python merge_colmap.py --train_dirs {source_target} {ext_source} --duplicate_resolution remaining"
            run_call(merge_call)
            os.makedirs(output, exist_ok=True)
            call = f"python msv2/train.py -s {source_target} -m {output} --start_checkpoint {checkpoint} {common_args_ext}"
            run_call(call)

        all_scenes.append(output)


if not args.skip_rendering:
    render_args = " --quiet --eval --skip_train "
    for scene in all_scenes:
        for save in save_iters:
            call = f"python render.py -m {scene} --iteration {save} {render_args}"
            run_call(call)

if not args.skip_metrics:
    for scene in all_scenes:
        call = f"python metrics.py -m {scene}"
        run_call(call)

