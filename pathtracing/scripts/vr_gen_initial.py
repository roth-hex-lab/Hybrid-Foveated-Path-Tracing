import os
import sys
import time
import datetime
import json
import math
import numpy as np
from scipy.stats import qmc
# import colmap import/export script
sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap
# import renderer module
import volpy


# helper functions
def sample_unit_sphere(sample):    
    z = 1.0 - 2.0 * sample[0]
    r = math.sqrt(max(0.0, 1.0 - z * z))
    phi = 2.0 * math.pi * sample[1]
    return volpy.vec3(r * math.cos(phi), r * math.sin(phi), z)


# Sample is a 3d sample from a normal distribution
def sample_inside_unit_sphere(sample, scale):
    x = sample[0]
    y = sample[1]
    z = sample[2]
    magnitude = math.sqrt(x*x + y*y + z*z)
    x /= magnitude
    y /= magnitude
    z /= magnitude

    s = math.cbrt(scale)
    return volpy.vec3(x*s, y*s, z*s)


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
    GEN_GT = False

    N_VIEWS = 12   # 12 16 32 64
    SAMPLES = 16   # 8 16 32 64
    BOUNCES = 4
    FOVY = 20
    PC_SIZE = 20000

    MAX_VARIABILITY = 1.0
    VARIABILITY_INC = 0.0
    
    FOV_RENDER = FOVY  
    FOV_COLMAP = 70  # FOV that works well with Gaussian Splatting TODO: Figure out whats going on

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, f'generated/TESTEST/leg-muscle/{N_VIEWS}p-{SAMPLES}s_walk-three')
    VOLUME = os.path.join(ROOT_DIR, 'data/Leg/IMG0001.dcm')
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    TRANSFER_FUNC = "data/lut_leg/muscle_gen.txt"
    OUT_FORMAT = ".txt"

    POINTCLOUD = False # Create initial pointcloud 

    DENOISE = True
    DENOISE_TEMPORAL = False
    GEN_ALBEDO = True
    GEN_MOTIONVEC = False
    GEN_DEPTH = False

    # Needed for albedo and depth generation
    # Don't use this for high sample counts
    TRACE_MULTI = True 
    
    WINDOW_LEFT = 0.0
    WINDOW_WIDTH = 1.0
    CUTOFF = 0.2

    ENV_ROTATION = 0.0
    ENV_ROT_AXIS = volpy.vec3(1,0,0)
    #ENV_ROTATION = 0.5 * math.pi        # 90°
    #ENV_ROT_AXIS = volpy.vec3(1, 0, 0)  # Rotation around X axis

    # No adjustments for leg
    #SCALE_ADJUST_Y = 0.0006
    #SCALE_ADJUST_Z = 0.0004

    DENSITY_SCALE = 1500
    ENV_STRENGTH = 2

    POS_ADJUST = 0 # Adjust camera position distance to the volume center
    FILENAME_PREFIX = ""

    ALBEDO = volpy.vec3(0.9, 0.9, 0.9)
    PHASE = 0
    SEED = 42
    BACKGROUND = False
    TONEMAPPING = True

    #Cut?
    CLIP_MIN = volpy.vec3(0, 0, 0)


    if GEN_GT:
        OUT_PATH += "_gt"
        FILENAME_PREFIX = "GT_"
        N_VIEWS = 32
        MAX_VARIABILITY = 0.7
        VARIABILITY_INC = 0.05 # More variability for GT
        POS_ADJUST = -2.5 # Closer to the volume
        SAMPLES = 4096
        BOUNCES = 100
        SEED = 777 # huh lucky!
        DENOISE = True
        GEN_ALBEDO = False
        GEN_MOTIONVEC = False
        GEN_DEPTH = False
        POINTCLOUD = False
        TRACE_MULTI = False
        

    # ------------------------------------------
    # Render colmap dataset

    # init
    renderer = volpy.Renderer()
    renderer.resize((2048,1024))
    renderer.init()

    if DENOISE:
        renderer.denoising_enabled = DENOISE
        renderer.denoise_temporal_enabled = DENOISE_TEMPORAL
        renderer.generate_motionvec = GEN_MOTIONVEC
        renderer.generate_albedo = GEN_ALBEDO
        renderer.generate_depth = GEN_DEPTH
        renderer.recreate_denoiser()

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
    renderer.density_scale = DENSITY_SCALE
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


    # Foveation stuff
    renderer.use_foveation = True
    renderer.fovea_deg = 7
    renderer.fovea_mult = 2
    renderer.fovea_center = volpy.vec2(0,0)
    unfoveate = False



    cameras = {}
    images = {}
    points3D = {}

    # HACK: write world-space AABB of volume as point3D (pos + rgb) to dataset
    points3D[0] = colmap.Point3D(id=0, xyz=np.array(renderer.volume.AABB("density")[0]), rgb=np.array(renderer.volume.AABB("density")[1]), error=0, image_ids=np.array([]), point2D_idxs=np.array([]))

    focal_original = renderer.colmap_focal_length()
    focal_adjusted = focal_original
    # Adjust focal length for colmap
    focal_adjusted = focal_original * (math.tan(math.radians(FOV_COLMAP/2)) / math.tan(math.radians(FOV_RENDER/2)))

    print(f"Focal length adjusted from {focal_original:.2f} to {focal_adjusted:.2f} for COLMAP compatibility.")

    # write camera
    cameras[0] = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([focal_adjusted, renderer.resolution().x//2, renderer.resolution().y//2]))

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    rng = np.random.default_rng(1)

    startTime = time.time()
    startTimeAbs = startTime

    cam_positions = []

    img_path = os.path.join(OUT_PATH, "images")
    os.makedirs(img_path, exist_ok=True)

    variability = 0

    bb_min, bb_max = renderer.volume.AABB("density")
    center = bb_min + (bb_max - bb_min) * 0.5
    radius = (bb_max - center).length()

    # Look at random point inside bounding box
    cam_target = center + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.8 * variability)            # Cam target somewhere withoing 80% radius of volume, increasing variability
    cam_unit_sphere_pos = center + sample_unit_sphere(samplerOut.random()[0, 0:2]) * radius
    cam_pos_moved_out = (cam_unit_sphere_pos + ((cam_unit_sphere_pos - center).normalize() * radius * (2 - variability + POS_ADJUST)))         # Cam position start point from 2.1 times radius down to 1.1 times radius around center
    cam_pos = cam_pos_moved_out + (sample_inside_unit_sphere(rng.standard_normal(3), rng.random()) * radius * 0.4 * variability)    # randomize camera position by moving up to 0.4 times radius around
    # write views
    for i in range(N_VIEWS):
        # setup camera

        variability = min(MAX_VARIABILITY, variability + VARIABILITY_INC)

        cam_positions.append([cam_pos.x, cam_pos.y, cam_pos.z])

        renderer.cam_pos = cam_pos
        renderer.cam_dir = (cam_target - renderer.cam_pos).normalize()
        renderer.cam_fov = FOV_RENDER

        fovea_x = i * -0.1
        renderer.fovea_center = volpy.vec2(fovea_x,0)
        # render view

        if i == 0:
            renderer.reset()

        if TRACE_MULTI:
            renderer.render_multi(SAMPLES)
        else:
            renderer.render(SAMPLES)
        
        renderer.process(do_unfoveate=unfoveate)

        # store view
        filename = f"{FILENAME_PREFIX}view_{i:06}_{fovea_x}.png"
        renderer.save(os.path.join(img_path, filename), silent = True)
        images[i] = colmap.Image(id=i, qvec=np.array(renderer.colmap_view_rot())[[3, 0, 1, 2]], tvec=np.array(renderer.colmap_view_trans()), camera_id=0, name=filename, xys=np.array([]), point3D_ids=np.array([]))



        # DELETE ME JUST TESTING
        if TRACE_MULTI:
            renderer.render_multi(SAMPLES)
        else:
            renderer.render(SAMPLES)
        
        renderer.process(do_unfoveate=True)

        # store view
        filename = f"{FILENAME_PREFIX}view_{i:06}_{fovea_x}_unfov.png"
        renderer.save(os.path.join(img_path, filename), silent=True)


        currentTime = time.time()
        diff = currentTime - startTime
        remaining = str(datetime.timedelta(seconds = ((N_VIEWS-i-1) * diff)))
        #print(f"Current frame took: {diff:.2f}s, overall: {remaining.split('.')[0]} left")
        startTime = currentTime




    print('--------------------')

    info_path = os.path.join(OUT_PATH, "sparse", "0")
    os.makedirs(info_path, exist_ok=True)
    colmap.write_model(cameras, images, points3D, path=info_path, ext=OUT_FORMAT)

    # Create initial pointcloud
    if POINTCLOUD:
        pc_path = os.path.join(info_path, "points3D.txt")
        os.rename(pc_path, os.path.join(info_path, "points3D.txt") + "._orig.txt")
        renderer.to_pointcloud_gpu(PC_SIZE, 128, pc_path)
        print('--------------------')


    # Write settings to folder too
    with open(os.path.join(info_path, 'settings.json'), 'w', encoding='utf-8') as file:
        data = dict(CAM_POSITIONS=cam_positions, VOLUME=VOLUME, ENVMAP=ENVMAP, SAMPLES=SAMPLES,TRANSFER_FUNC=TRANSFER_FUNC, N_VIEWS=N_VIEWS, WINDOW_LEFT=WINDOW_LEFT, WINDOW_WIDTH=WINDOW_WIDTH, CUTOFF=CUTOFF, ENV_STRENGTH=ENV_STRENGTH, CLIP_MIN=str(CLIP_MIN))
        json.dump(data, file, ensure_ascii=False, indent=4)

    diff = time.time() - startTimeAbs
    print(f"Done, took {str(datetime.timedelta(seconds = diff)).split('.')[0]}")
