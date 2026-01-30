import math
from pathlib import Path

# This assumes the script is located in scripts/vr_gen_config.py, and the root should be one parent directory up.
ROOT_DIR = Path(__file__).resolve().parent.parent

# These settings are considered universal and are inherited by all configurations.
general_settings = {
    "ALBEDO": [0.9, 0.9, 0.9],
    "PHASE": 0,
    "BACKGROUND": False,
    "TONEMAPPING": True,
}

# Settings for the offline multi-view generation script (vr_gen_initial_multi.py)
offline_renderer_config = {
    "OUT_PATH": ROOT_DIR / "generated" / "vrinit_multi_comp",
    "N_VIEWS": [],
    "SAMPLES": [],
    "RESOLUTION": (512, 512),
    "MULTIPLIER": 5,
    "DRY_RUN_REPETITIONS": 0,
    "GEN_GT": True,
    "BOUNCES": 4,
    "PC_SIZE": 20000,
    "MAX_VARIABILITY": 1.7,
    "VARIABILITY_INC": 0.04,
    "RETRY_EMPTY_VIEWS": True,
    "EMPTY_VIEW_ALPHA_THRESHOLD": 0.04,
    "RETRY_TOO_CLOSE": True,
    "TOO_CLOSE_THRESHOLD_UNITS": 0.05,
    "TOO_CLOSE_FRACTION": 0.2,
    "DENOISE": True,
    "DENOISE_TEMPORAL": False,
    "GEN_ALBEDO": True,
    "GEN_MOTIONVEC": False,
    "GEN_DEPTH": True,
    "TRACE_MULTI": True,
    "FOV_RENDER": 20,
    "SEED": 42,
    "COMPARISON": {
        "FILENAME_PREFIX": "COMP_",
        "N_VIEWS": [256],
        "SAMPLES": [512],
        "RESOLUTION": (1024, 1024),
        "BOUNCES": 100,
        "MAX_VARIABILITY": 2.5,
        "VARIABILITY_INC": 0.02,
        "POS_ADJUST": -1,
        "SEED": 20,
        "DENOISE": True,
        "GEN_ALBEDO": False,
        "GEN_MOTIONVEC": False,
        "TRACE_MULTI": True
    }
}

# Settings for the real-time render server (vr_gen_server.py)
server_config = {
    "HOST": "localhost",
    "PORT": 9999,
    "DATASET_TO_USE": "leg_muscle",  # Specifies which dataset to load on startup
    "RESOLUTION": (512, 512),
    "SAMPLES": 16,
    "BOUNCES": 4,
    "DENOISE": True,
    "DENOISE_TEMPORAL": True,
    "GEN_ALBEDO": True,
    "GEN_MOTIONVEC": True,
    "GEN_DEPTH": True,
    "FOV_RENDER": 20,
    "SEED": 42,
    "STORE_RENDER_BUFFER": True, # For passing along to periphery model refinement
    "MAX_BUFFER_SIZE": 100,
    "REJECTION_SAMPLING_ENABLED": True,
    "REJECTION_DISTANCE_THRESHOLD": 0.1,  # Minimum distance between camera positions
    "REJECTION_ANGLE_THRESHOLD": 5.0      # Minimum angle in degrees between camera directions
}


# These settings will override any of the above settings.
datasets = {
    "fullbody_bone": {
        "VOLUME": ROOT_DIR / 'data/Fullbody-nobed-cropped/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_fullbody/bone.txt",
        "WINDOW_LEFT": 0.0,
        "WINDOW_WIDTH": 1.0,
        "CUTOFF": 0.275,
        "ENV_STRENGTH": 2,
        "SCALE_ADJUST_Y": 0.00057,
        "SCALE_ADJUST_Z": 0.00037,
        "ENV_ROTATION": 0.5 * math.pi,
        "ENV_ROT_AXIS": [1, 0, 0],
    },
    "fullbody_organs": {
        "VOLUME": ROOT_DIR / 'data/Fullbody-nobed-cropped/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_fullbody/organs.txt",
        "WINDOW_LEFT": 0.0,
        "WINDOW_WIDTH": 1.0,
        "CUTOFF": 0.275,
        "ENV_STRENGTH": 2,
        "SCALE_ADJUST_Y": 0.00057,
        "SCALE_ADJUST_Z": 0.00037,
        "ENV_ROTATION": 0.5 * math.pi,
        "ENV_ROT_AXIS": [1, 0, 0],
    },
    "fullbody_muscle": {
        "VOLUME": ROOT_DIR / 'data/Fullbody-nobed-cropped/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_fullbody/muscle.txt",
        "WINDOW_LEFT": 0.0015,
        "WINDOW_WIDTH": 1.0,
        "CUTOFF": 0.275,
        "ENV_STRENGTH": 2,
        "SCALE_ADJUST_Y": 0.00057,
        "SCALE_ADJUST_Z": 0.00037,
        "ENV_ROTATION": 0.5 * math.pi,
        "ENV_ROT_AXIS": [1, 0, 0],
    },
    "leg_bone": {
        "VOLUME": ROOT_DIR / 'data/Leg/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_leg/bone_gen.txt",
        "WINDOW_LEFT": 0.0,
        "WINDOW_WIDTH": 1.0,
        "CUTOFF": 0.2,
        "ENV_STRENGTH": 2
    },
    "leg_muscle": {
        "VOLUME": ROOT_DIR / 'data/Leg/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_leg/muscle_gen.txt",
        "WINDOW_LEFT": 0.005,
        "WINDOW_WIDTH": 1.0,
        "CUTOFF": 0.3,
        "ENV_STRENGTH": 2,
    },
    "leg_full": {
        "VOLUME": ROOT_DIR / 'data/Leg/IMG0001.dcm',
        "ENVMAP": ROOT_DIR / 'data/table_mountain_2_puresky_1k.hdr',
        "TRANSFER_FUNC": ROOT_DIR / "data/lut_leg/full_gen_two.txt",
        "WINDOW_LEFT": 0.013,
        "WINDOW_WIDTH": 0.996,
        "CUTOFF": 0.35,
        "ENV_STRENGTH": 2,
        "DENSITY_SCALE": 1500,
    },
}