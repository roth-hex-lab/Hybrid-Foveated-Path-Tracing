import os
from argparse import ArgumentParser
import shutil
import subprocess
import sys

datasets = ["leg_bone", "leg_muscle", "leg_full", "fullbody_bone", "fullbody_organs", "fullbody_muscle"]
pic_variants = [16]
sample_variants = [8]
trials = 1
save_iters = [700]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./eval/init_compare/")
parser.add_argument("--data-root")

args, _ = parser.parse_known_args()

all_scenes = []

def run_call(call: str) -> None:
    try:
        subprocess.run(call, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}): {call}", file=sys.stderr)
        sys.exit(e.returncode)


common_args = " --quiet --eval --test_iterations -1 --config_path config/anatomy_fast "
common_args += f" --test_iterations -1 --save_iterations {' '.join(map(str, save_iters))} "

for dataset in datasets:
    for pic in pic_variants:
        for sample in sample_variants:
            for trial in range(trials):
                id_fragment = os.path.join(dataset, "train", f"{pic}p_{sample}s_{trial}")
                source = os.path.join(args.data_root, id_fragment)
                output = os.path.join(args.output_path, id_fragment)

                if not args.skip_training:
                    os.makedirs(output, exist_ok=True)
                    call = f"python msv2/train.py -s {source} -m {output} {common_args}"
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

