import os
import sys
import time
import datetime
import json
import math
import random
import numpy as np
from scipy.stats import qmc
# import colmap import/export script
sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap
# import renderer module
import volpy


# helper functions
def sample_unit_sphere(sample):
    # Somehow have to import again for pybind11?
    import math
    import volpy
    
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)


# Sample is a 3d sample from a normal distribution
def sample_inside_unit_sphere(sample):
    # Somehow have to import again for pybind11?
    import math
    import random
    import volpy

    x = sample[0]
    y = sample[1]
    z = sample[2]
    magnitude = math.sqrt(x*x + y*y + z*z)
    x /= magnitude
    y /= magnitude
    z /= magnitude
    s = math.cbrt(random.random())
    return volpy.vec3(x*s, y*s, z*s)


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    N_VIEWS = 32 
    SAMPLES = 16
    BOUNCES = 4
    FOVY = 20
    
    FOV_RENDER = FOVY  # Your desired narrow FOV for rendering
    FOV_COLMAP = 70  # FOV that works well with Gaussian Splatting

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, f'generated/speedtest/vr-{N_VIEWS}p-{SAMPLES}s_{FOVY}fov')
    #VOLUME = os.path.join(ROOT_DIR, 'data/Fullbody-nobed-cropped/IMG0001.dcm')
    VOLUME = os.path.join(ROOT_DIR, 'data/Leg/IMG0001.dcm')
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    TRANSFER_FUNC = "data/lut_leg/muscle_gen_two.txt"
    OUT_FORMAT = ".txt"

    POINTCLOUD = False # Create initial pointcloud 
    PC_SIZE = 15
    DENOISE = True
    
    WINDOW_LEFT = 0.0
    WINDOW_WIDTH = 1.0
    CUTOFF = 0.275
    ENV_ROTATION = 0.5 * math.pi        # 90°
    ENV_ROT_AXIS = volpy.vec3(1, 0, 0)  # Rotation around X axis

    # No adjustments for leg
    #SCALE_ADJUST_Y = 0.0006
    #SCALE_ADJUST_Z = 0.0004

    #DENSITY_SCALE = 1500
    ENV_STRENGTH = 2

    ALBEDO = volpy.vec3(0.9, 0.9, 0.9)
    PHASE = 0
    SEED = 42
    BACKGROUND = False
    TONEMAPPING = True

    #Cut?
    CLIP_MIN = volpy.vec3(0, 0, 0)

    # ------------------------------------------
    # Render colmap dataset

    # init
    renderer = volpy.Renderer()
    renderer.init()
    renderer.draw()
    OUT_PATH += f"_{renderer.resolution().x}x{renderer.resolution().y}"
    os.makedirs(OUT_PATH, exist_ok=True)

    # setup scene
    renderer.volume = volpy.Volume(VOLUME)
    renderer.scale_and_move_to_unit_cube()
    renderer.commit()
    try:
        renderer.volume.transform.set_value(1, 1, SCALE_ADJUST_Y)
    except NameError:
        pass
    try:
        renderer.volume.transform.set_value(2, 2, SCALE_ADJUST_Z)
    except NameError:
        pass

    renderer.seed = SEED
    renderer.bounces = BOUNCES
    renderer.albedo = ALBEDO
    renderer.phase = PHASE
    #renderer.density_scale = DENSITY_SCALE
    renderer.environment = volpy.Environment(ENVMAP)
    renderer.environment.strength = ENV_STRENGTH
    renderer.show_environment = BACKGROUND
    renderer.tonemapping = TONEMAPPING
    renderer.vol_clip_min = CLIP_MIN
    renderer.cutoff = CUTOFF

    renderer.rotate_env(ENV_ROTATION, ENV_ROT_AXIS)

    tf = volpy.TransferFunction(TRANSFER_FUNC)
    tf.window_left = WINDOW_LEFT
    tf.window_width = WINDOW_WIDTH
    renderer.transferfunc = tf
    renderer.denoising_enabled = DENOISE

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    rng = np.random.default_rng()

    startTime = time.time()
    startTimeAbs = startTime

    rendertimes = []

    # write views
    for i in range(N_VIEWS):
        # setup camera
        bb_min, bb_max = renderer.volume.AABB("density")
        center = bb_min + (bb_max - bb_min) * 0.5
        radius = (bb_max - center).length()


        # Look at random point inside bounding box
        cam_target = center + (sample_inside_unit_sphere(rng.standard_normal(3)) * radius * 0.7)                      # Cam target somewhere withoing 80% radius of volume, increasing variability
        cam_unit_sphere_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
        cam_pos_moved_out = (cam_unit_sphere_pos + ((cam_unit_sphere_pos - center).normalize() * radius * 1.1)) 
        cam_pos = cam_pos_moved_out + (sample_inside_unit_sphere(rng.standard_normal(3)) * radius * 0.3)

        startTime = time.time()


        renderer.cam_pos = cam_pos
        renderer.cam_dir = (cam_target - renderer.cam_pos).normalize()
        renderer.cam_fov = FOV_RENDER
        # render view
        renderer.render(SAMPLES)
        renderer.draw()
        # store view


        currentTime = time.time()
        diff = currentTime - startTime
        rendertimes.append(diff)



    print('--------------------')
    np_data = np.array(rendertimes, dtype=np.float32)
    time_avg = np.mean(np_data)
    time_std = np.std(np_data)
    print(f'Average render time: {time_avg:.3f} s, std: +-{time_std:.3f} s')


    # Create initial pointcloud
    if POINTCLOUD:
        renderer.to_pointcloud(PC_SIZE, OUT_PATH)
        print('--------------------')


    # Write settings to folder too
    with open(os.path.join(OUT_PATH, 'settings.json'), 'w', encoding='utf-8') as file:
        data = dict(VOLUME=VOLUME, ENVMAP=ENVMAP, SAMPLES=SAMPLES,TRANSFER_FUNC=TRANSFER_FUNC, N_VIEWS=N_VIEWS, WINDOW_LEFT=WINDOW_LEFT, WINDOW_WIDTH=WINDOW_WIDTH, CUTOFF=CUTOFF, ENV_STRENGTH=ENV_STRENGTH, CLIP_MIN=str(CLIP_MIN), times=rendertimes, time_avg=str(time_avg), time_std=str(time_std))
        json.dump(data, file, ensure_ascii=False, indent=4)

    diff = time.time() - startTimeAbs
    print(f"Done, took {str(datetime.timedelta(seconds = diff)).split('.')[0]}")

    renderer.shutdown()
