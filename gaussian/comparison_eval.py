import os
from argparse import ArgumentParser
import subprocess
import sys

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip-gt-gs", action="store_true")
parser.add_argument("--skip-gt-ptf", action="store_true")
parser.add_argument("--skip-ours", action="store_true")
parser.add_argument("--data-root")

args, _ = parser.parse_known_args()

all_scenes = []

def run_call(call: str) -> None:
    try:
        subprocess.run(call, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode}): {call}", file=sys.stderr)
        sys.exit(e.returncode)

# subpic 504 = 20° deg fov cutout of a 70° fov 200px square img

if not args.skip_gt_gs:
    call = f"python metrics.py -m {os.path.join(args.data_root, 'gt_gs')}  --subpic 504"
    run_call(call)

if not args.skip_gt_ptf:
    call = f"python metrics.py -m {os.path.join(args.data_root, 'gt_pathtracing_foveated')}  --subpic 504"
    call_2 = f"python metrics.py -m {os.path.join(args.data_root, 'gt_pt_fov2')}  --subpic 504"
    #run_call(call)
    run_call(call_2)

if not args.skip_ours:
    call = f"python metrics.py -m {os.path.join(args.data_root, 'ours')} --subpic 504" 
    #run_call(call)

    call = f"python metrics.py -m {os.path.join(args.data_root, 'ours_5inc')} --subpic 504" 
    #run_call(call)

    call = f"python metrics.py -m {os.path.join(args.data_root, 'ours_10inc')} --subpic 504" 
    #run_call(call)

    call = f"python metrics.py -m {os.path.join(args.data_root, 'ours_15inc')} --subpic 504" 
    #run_call(call)

    call = f"python metrics.py -m {os.path.join(args.data_root, 'ours_420')} --subpic 504" 
    run_call(call)

