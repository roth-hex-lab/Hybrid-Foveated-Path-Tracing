import os
import sys
import time
import datetime
import json
import math
import numpy as np
import shutil
import csv
from pathlib import Path
from scipy.stats import qmc
from typing import Optional, Dict, Any, List

# Ensure the script can find the colmap and renderer modules
sys.path.append(str(Path(__file__).parent))
import read_write_model as colmap
import volpy

from vr_gen_config_foveated_ext import general_settings, offline_renderer_config, datasets

def sample_unit_sphere(sample: np.ndarray) -> volpy.vec3:
    """Generates a random point on the surface of a unit sphere."""
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)


def sample_inside_unit_sphere(sample: np.ndarray, scale: float) -> volpy.vec3:
    """Generates a random point inside a unit sphere with a given scale."""
    x, y, z = sample[0], sample[1], sample[2]
    magnitude = math.sqrt(x*x + y*y + z*z)
    x /= magnitude
    y /= magnitude
    z /= magnitude
    s = math.cbrt(scale)
    return volpy.vec3(x*s, y*s, z*s)


# --- Core Logic Functions ---

def setup_renderer(config: Dict[str, Any]) -> volpy.Renderer:
    """Initializes and configures a new renderer instance based on the provided settings."""
    renderer = volpy.Renderer()
    override_res = config.get("RESOLUTION")
    if override_res:
        renderer.resize(override_res)
    renderer.init()

    if config.get("DENOISE"):
        renderer.denoising_enabled = True
        renderer.denoise_temporal_enabled = config.get("DENOISE_TEMPORAL", False)
        renderer.generate_motionvec = config.get("GEN_MOTIONVEC", False)
        renderer.generate_albedo = config.get("GEN_ALBEDO", False)
        renderer.generate_depth = config.get("GEN_DEPTH", False)
        renderer.recreate_denoiser()

    renderer.volume = volpy.Volume(str(config["VOLUME"]))
    renderer.scale_and_move_to_unit_cube()
    renderer.commit()

    if config.get("FOVEATE"):
        renderer.use_foveation = True
        renderer.fovea_deg = config.get("FOVEATE_DEG", 7)
        renderer.fovea_mult = config.get("FOVEATE_MULT", 2)

    if config.get("SCALE_ADJUST_Y") is not None:
        renderer.volume.transform.set_value(1, 1, config["SCALE_ADJUST_Y"])
    if config.get("SCALE_ADJUST_Z") is not None:
        renderer.volume.transform.set_value(2, 2, config["SCALE_ADJUST_Z"])

    renderer.seed = config.get("SEED", 42)
    renderer.bounces = config.get("BOUNCES", 100)
    if config.get("ALBEDO"): renderer.albedo = volpy.vec3(*config["ALBEDO"])
    if config.get("PHASE"): renderer.phase = config["PHASE"]
    if config.get("DENSITY_SCALE"): renderer.density_scale = config["DENSITY_SCALE"]
    
    if config.get("ENVMAP"):
        renderer.environment = volpy.Environment(str(config["ENVMAP"]))
        renderer.environment.strength = config.get("ENV_STRENGTH", 1.0)
    
    renderer.show_environment = config.get("BACKGROUND", False)
    renderer.tonemapping = config.get("TONEMAPPING", True)
    if config.get("CLIP_MIN"): renderer.vol_clip_min = volpy.vec3(*config["CLIP_MIN"])
    renderer.cutoff = config.get("CUTOFF", 0.0)

    if config.get("ENV_ROTATION") and config.get("ENV_ROT_AXIS"):
        renderer.rotate_env(config["ENV_ROTATION"], volpy.vec3(*config["ENV_ROT_AXIS"]))

    tf = volpy.TransferFunction(str(config["TRANSFER_FUNC"]))
    tf.window_left = config.get("WINDOW_LEFT", 0.0)
    tf.window_width = config.get("WINDOW_WIDTH", 1.0)
    renderer.transferfunc = tf

    return renderer


def render_view_set(
    renderer: volpy.Renderer, 
    config: Dict[str, Any], 
    n_views: int, 
    samples: int,
    cam_pos_data: tuple, 
    output_path: Optional[Path] = None, 
    filename_prefix: str = ""
):
    """Renders a set of views, saves them if a path is provided, and returns per-frame timings."""
    images, cam_positions, frame_times = {}, [], []
    variability = 0.0
    max_variability = config.get("MAX_VARIABILITY", 1.0)
    variability_inc = config.get("VARIABILITY_INC", 0.02)
    pos_adjust = config.get("POS_ADJUST", 0.0)
    samplerOut, rng = cam_pos_data

    # This check will only be active during actual image generation (not dry run)
    perform_empty_check = (output_path is not None) and config.get("RETRY_EMPTY_VIEWS", False)
    alpha_threshold = config.get("EMPTY_VIEW_ALPHA_THRESHOLD", 0.01)
    perform_too_close_check = (output_path is not None) and config.get("RETRY_TOO_CLOSE", False)
    max_retries = config.get("MAX_RETRIES_PER_VIEW", 200)
    
    if output_path: print(f"    - Rendering and saving {n_views} views with {samples} samples...")
    
    total_start_time = time.perf_counter()
    for i in range(n_views):
        is_view_valid = False
        retry_count = 0
        if config.get("RANDOM_FOVEATIONS", False):
            # do not use existing random here so positions stay consitent with existing images
            import random
            rng_foveation = random.Random(i)
    
            # Approximate eye positions with circular distribution skewing center
            angle = rng_foveation.uniform(0, 2 * math.pi)
            radius = rng_foveation.triangular(0, 0.4, 0)
            fovea_x = radius * math.cos(angle)
            fovea_y = radius * math.sin(angle)

            renderer.fovea_center = volpy.vec2(fovea_x,fovea_y)

        while not is_view_valid and retry_count < max_retries:
            frame_start_time = time.perf_counter()
            
            # --- Camera generation logic (moved inside the loop) ---
            bb_min, bb_max = renderer.volume.AABB("density")
            center = bb_min + (bb_max - bb_min) * 0.5
            radius = (bb_max - center).length()
            cam_target = center + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.8 * variability)
            cam_unit_sphere_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
            cam_pos_moved_out = (cam_unit_sphere_pos + ((cam_unit_sphere_pos - center).normalize() * radius * (4.5 - variability + pos_adjust)))
            cam_pos = cam_pos_moved_out + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.4 * variability)
            
            renderer.cam_pos, renderer.cam_dir, renderer.cam_fov = cam_pos, (cam_target - cam_pos).normalize(), config["FOV_RENDER"]
            
            if i == 0 and retry_count == 0: renderer.reset()

            # --- Render the view ---
            trace_multi = config.get("TRACE_MULTI", True)
            if trace_multi:
                renderer.render_multi(samples)
            else:
                renderer.render(samples)
            
            unfoveate = config.get("UNFOVEATE", False)
            renderer.process(do_unfoveate=unfoveate)
            frame_end_time = time.perf_counter()

            too_empty = False
            if perform_empty_check:
                rgba_data = np.array(renderer.rgba_data())
                # Select every 4th element, starting from index 3 (rgba_data is 1 dimensional)
                alpha_channel = rgba_data[3::4]
                average_alpha = float(np.mean(alpha_channel)) / 255.0

                too_empty = average_alpha < alpha_threshold

            too_close = False
            if perform_too_close_check and renderer.generate_depth and trace_multi:
                depth_raw = np.array(renderer.depth_data(), dtype=np.float32)  # 1D, size=W*H
                # Depth==0 means "no hit" (we ignore those for this statistic)
                valid = depth_raw > 0.0

                if np.any(valid):
                    vol_scale = renderer.volume.transform.value(0, 0) * 1000 # Assumes uniform scale, and units in mm
                    vol_scale = max(min(vol_scale, 2), 0.5) # Keep some bounds as we dont know what units were operating with
                    dist_units = depth_raw * float(config.get("DEPTH_INV_SCALE", 4.0)) / float(vol_scale)

                    thr = float(config.get("TOO_CLOSE_THRESHOLD_UNITS", 0.05))
                    frac = float(config.get("TOO_CLOSE_FRACTION", 0.30))

                    frac_close = float(np.mean((dist_units < thr) & valid))
                    too_close = frac_close > frac

            if too_empty or too_close:
                retry_count += 1
            else:
                is_view_valid = True


        # --- End of retry loop ---

        if retry_count >= max_retries:
            print(f"      - WARNING: Max retries reached for view {i}. Accepting potentially empty view.")

        # --- Store results for the now-validated view ---
        frame_times.append(frame_end_time - frame_start_time)
        cam_positions.append([cam_pos.x, cam_pos.y, cam_pos.z])
        variability = min(max_variability, variability + variability_inc)
        
        filename = f"{filename_prefix}view_{i:06}.png"
        if output_path: renderer.save(str(output_path / filename), silent=True)
        images[i] = colmap.Image(id=i, qvec=np.array(renderer.colmap_view_rot())[[3, 0, 1, 2]], tvec=np.array(renderer.colmap_view_trans()), camera_id=0, name=filename, xys=np.array([]), point3D_ids=np.array([]))
    
    total_duration = time.perf_counter() - total_start_time
    if output_path: print(f"      Done in {total_duration:.2f}s.")
    return images, cam_positions, frame_times


def generate_point_cloud(renderer: volpy.Renderer, config: Dict[str, Any], dataset_path: Path) -> Path:
    """Generates the point cloud using a pre-configured renderer instance."""
    print("  - Generating point cloud (reusing renderer)...")
    pc_start_time = time.perf_counter()
    pc_size, temp_pc_path = config.get("PC_SIZE", 20000), dataset_path / "points3D.txt.tmp"
    renderer.commit()
    renderer.to_pointcloud_gpu(pc_size, 64, str(temp_pc_path))
    print(f"    Point cloud generated in {time.perf_counter() - pc_start_time:.2f}s.")
    return temp_pc_path


def generate_ground_truth(config: Dict[str, Any], dataset_path: Path) -> None:
    """Generates high-quality ground truth images."""
    if not config.get("GEN_GT", False):
        return

    print("  - Generating ground truth images...")
    gt_config = {**config, **config.get("GT_SETTINGS", {})}
    renderer = setup_renderer(gt_config)
    run_path = dataset_path / "GT"
    output_path = run_path / "images"
    output_path.mkdir(parents=True, exist_ok=True)

    sampler = (qmc.Sobol(d=2, rng=gt_config["SEED"] + 1), np.random.default_rng(gt_config["SEED"]))
    images, cam_pos, _ = render_view_set(renderer, gt_config, gt_config["N_VIEWS"], gt_config["SAMPLES"], sampler, output_path=output_path, filename_prefix=gt_config.get("FILENAME_PREFIX", "GT_"))
    save_run_artifacts(run_path, renderer, images, cam_pos, gt_config)

    print("    Ground truth generation complete.")


def save_run_artifacts(run_path: Path, renderer: volpy.Renderer, images: Dict, cam_pos: list, config_to_save: Dict, pc_source_path: Path | None = None):
    """Saves COLMAP data, point cloud, and settings file for a run."""
    info_path = run_path / "sparse" / "0"
    info_path.mkdir(parents=True, exist_ok=True)

    # Save COLMAP model
    focal = renderer.colmap_focal_length()
    if "FOV_COLMAP" in config_to_save:
        focal = focal * (math.tan(math.radians(config_to_save["FOV_COLMAP"]/2)) / math.tan(math.radians(config_to_save["FOV_RENDER"]/2)))
        
    cam = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([focal, renderer.resolution().x//2, renderer.resolution().y//2]))
    p3D = colmap.Point3D(id=0, xyz=np.array(renderer.volume.AABB("density")[0]), rgb=np.array(renderer.volume.AABB("density")[1]), error=0, image_ids=np.array([]), point2D_idxs=np.array([]))
    colmap.write_model({0: cam}, images, {0: p3D}, path=str(info_path), ext=".txt")

    # Copy point cloud
    if pc_source_path:
        shutil.copy(pc_source_path, info_path / "points3D.txt")
    run_config = {**config_to_save, "CAM_POSITIONS": cam_pos}
    with open(run_path / 'settings.json', 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=4, default=str)


def process_dataset(dataset_name: str, config: Dict[str, Any]) -> None:
    """Main function to process a single dataset configuration."""
    dataset_start_time = time.perf_counter()
    print(f"\n--- Processing dataset: {dataset_name} ---")
    for key in ["VOLUME", "ENVMAP", "TRANSFER_FUNC"]:
        if not Path(config[key]).exists():
            print(f"Error: Required file not found for key '{key}': {config[key]}", file=sys.stderr); return
    dataset_path = Path(config["OUT_PATH"]) / dataset_name
    dataset_path.mkdir(parents=True, exist_ok=True)

    # 1. Setup renderer for dry-run and point cloud
    print("  - Initializing renderer for dry run and point cloud generation...")
    shared_renderer = setup_renderer(config)

    # 2. Dry Run Logic
    print("  - Starting detailed dry run for performance timing...")
    DRY_RUN_REPETITIONS = config.get("DRY_RUN_REPETITIONS", 3)

    if DRY_RUN_REPETITIONS > 0:
        all_timings_data = []
        for n_views in config["N_VIEWS"]:
            for samples in config["SAMPLES"]:
                print(f"    - Benchmarking {n_views}p x {samples}s ({DRY_RUN_REPETITIONS} repetitions)...")
                repetition_results = []
                for rep in range(DRY_RUN_REPETITIONS):
                    total_run_start_time = time.perf_counter()
                    sampler = (qmc.Sobol(d=2, rng=config['SEED'] + 1), np.random.default_rng(config['SEED']))
                    _, _, frame_times = render_view_set(shared_renderer, config, n_views, samples, sampler)
                    total_run_duration = time.perf_counter() - total_run_start_time
                    
                    avg_ms = np.mean(frame_times) * 1000
                    std_dev_ms = np.std(frame_times) * 1000
                    repetition_results.append({
                        "avg_ms": avg_ms, "std_dev_ms": std_dev_ms, "total_s": total_run_duration
                    })
                    print(f"      - Rep {rep + 1}/{DRY_RUN_REPETITIONS}: Avg: {avg_ms:.3f}ms, StdDev: {std_dev_ms:.3f}ms")

                best_run = min(repetition_results, key=lambda x: x['avg_ms'])
                print(f"      - Best result: Avg: {best_run['avg_ms']:.3f}ms")
                all_timings_data.append({
                    'n_views': n_views, 'samples': samples,
                    'best_avg_ms': f"{best_run['avg_ms']:.3f}",
                    'std_dev_at_best_ms': f"{best_run['std_dev_ms']:.3f}",
                    'total_time_at_best_s': f"{best_run['total_s']:.3f}"
                })

        csv_path = dataset_path / 'render_speed.csv'
        fieldnames = ['n_views', 'samples', 'best_avg_ms', 'std_dev_at_best_ms', 'total_time_at_best_s']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_timings_data)
        print(f"    Performance timings saved to {csv_path}")

    # 3. Reuse the renderer for point cloud generation
    temp_pc_path = generate_point_cloud(shared_renderer, config, dataset_path)
    del shared_renderer

    # 4. Generate Ground Truth
    if "GT_SETTINGS" in config:
        generate_ground_truth(config, dataset_path)
    else:
        print("  - Skipping ground truth generation as GT_SETTINGS are not defined.")

    # 5. Main Rendering Loop
    print("\n  - Starting main rendering process...")
    main_renderer = setup_renderer(config)

    for mult in range(math.floor(config.get("MULTIPLIER", 1))):
        for n_views, samples in [(n, s) for n in config["N_VIEWS"] for s in config["SAMPLES"]]:
            run_path = dataset_path / "train" / f"{n_views}p_{samples}s_{mult}"
            output_path = run_path / "images"
            output_path.mkdir(parents=True, exist_ok=True)
            sampler = (qmc.Sobol(d=2, rng=config["SEED"] + 1 + mult), np.random.default_rng(config["SEED"] + mult))
            run_config = {**config, "N_VIEWS": n_views, "SAMPLES": samples}

            images, cam_pos, _ = render_view_set(main_renderer, run_config, n_views, samples, sampler, output_path)
            save_run_artifacts(run_path, main_renderer, images, cam_pos, run_config, temp_pc_path)
    
    del main_renderer
    
    # 6. Extensions Rendering Loop
    if "EXTENSIONS" in config:
        print("\n  - Starting extensions rendering process...")
        ext_config = {**config, **config.get("EXTENSIONS", {})}
        ext_renderer = setup_renderer(ext_config)

        for n_views, samples in [(n, s) for n in ext_config["N_VIEWS"] for s in ext_config["SAMPLES"]]:
            run_path = dataset_path / "ext" / f"{n_views}p_{samples}s_EXT"
            output_path = run_path / "images"
            output_path.mkdir(parents=True, exist_ok=True)
            sampler = (qmc.Sobol(d=2, rng=ext_config["SEED"] + 1), np.random.default_rng(ext_config["SEED"]))
            run_ext_config = {**ext_config, "N_VIEWS": n_views, "SAMPLES": samples}
            prefix = run_ext_config.get("FILENAME_PREFIX", "EXT_")

            images, cam_pos, _ = render_view_set(ext_renderer, run_ext_config, n_views, samples, sampler, output_path, filename_prefix=prefix)
            save_run_artifacts(run_path, ext_renderer, images, cam_pos, run_ext_config, temp_pc_path)
        
        del ext_renderer


    # 7. Comparison rendering
    if "COMPARISON" in config:
        print("\n  - Starting comparison rendering process...")
        comp_config = {**config, **config.get("COMPARISON", {})}
        comp_renderer = setup_renderer(comp_config)

        for n_views, samples in [(n, s) for n in comp_config["N_VIEWS"] for s in comp_config["SAMPLES"]]:
            run_path = dataset_path / "comp" / f"{n_views}p_{samples}s_COMP"
            output_path = run_path / "images"
            output_path.mkdir(parents=True, exist_ok=True)
            sampler = (qmc.Sobol(d=2, rng=comp_config["SEED"] + 1), np.random.default_rng(comp_config["SEED"]))
            run_comp_config = {**comp_config, "N_VIEWS": n_views, "SAMPLES": samples}
            prefix = run_comp_config.get("FILENAME_PREFIX", "COMP_")

            images, cam_pos, _ = render_view_set(comp_renderer, run_comp_config, n_views, samples, sampler, output_path, filename_prefix=prefix)
            save_run_artifacts(run_path, comp_renderer, images, cam_pos, run_comp_config, temp_pc_path)
        
        del comp_renderer

    # Final cleanup
    temp_pc_path.unlink()
    total_duration = time.perf_counter() - dataset_start_time
    print(f"--- Finished dataset '{dataset_name}' in {str(datetime.timedelta(seconds=total_duration)).split('.')[0]} ---")

if __name__ == "__main__":
    overall_start_time = time.perf_counter()
    
    for name, dataset_specific_config in datasets.items():
        final_config = {
            **general_settings, 
            **offline_renderer_config, 
            **dataset_specific_config
        }
        process_dataset(name, final_config)

    total_duration = time.perf_counter() - overall_start_time
    print(f"\nAll datasets processed. Total time: {str(datetime.timedelta(seconds=total_duration)).split('.')[0]}")