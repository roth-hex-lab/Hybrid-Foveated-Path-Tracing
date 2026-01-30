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


class RenderServer:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.client_socket = None
        self.running = False

    def start_server(self):
        import socket
        import threading

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        
        print(f"Listening on {self.host}:{self.port}")
        
        while True:
            client, addr = server.accept()
            print(f"Connection from {addr}")
            self.client_socket = client
            self.running = True
            
            # Start threads for sending/receiving
            threading.Thread(target=self.handle_client, daemon=True).start()
            
    def handle_client(self):
        import msgpack
        import struct

        while self.running:
            try:
                data = self.receive_message(struct)
                if data:
                    # Process pose data
                    pose = msgpack.unpackb(data)
                    print(f"Received pose: {pose}")
                    
                    # Send back rendered frame
                    self.send_frame()
                    
            except Exception as e:
                print(f"Client disconnected: {e}")
                self.running = False
                
    def receive_message(self, struct):
        # Read message length first (4 bytes)
        length_data = self.client_socket.recv(4)
        if not length_data:
            return None
        
        length = struct.unpack('!I', length_data)[0]
        
        # Read the actual message
        data = b''
        while len(data) < length:
            chunk = self.client_socket.recv(min(length - len(data), 4096))
            if not chunk:
                return None
            data += chunk
        
        return data
    
    def send_frame(self):
        import msgpack
        import struct

        # Generate frame here
        #frame_data = self.render_frame()
        
        # Create message with frame data
        message = msgpack.packb({
            'type': 'frame',
            'id': 0,
            'width': 1920,
            'height': 1080,
            'data': "frame_data"
        })
        
        # Send length first, then data
        length = len(message)
        self.client_socket.sendall(struct.pack('!I', length) + message)


if __name__ == "__main__":

    ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

    N_VIEWS = 16 
    SAMPLES = 64
    BOUNCES = 4
    FOVY = 20
    
    FOV_RENDER = FOVY  # Your desired narrow FOV for rendering
    FOV_COLMAP = 70  # FOV that works well with Gaussian Splatting

    # settings
    OUT_PATH = os.path.join(ROOT_DIR, f'generated/debug-vrinit_leg/vr-{N_VIEWS}p-{SAMPLES}s_{FOVY}fov_debug2')
    #VOLUME = os.path.join(ROOT_DIR, 'data/Fullbody-nobed-cropped/IMG0001.dcm')
    VOLUME = os.path.join(ROOT_DIR, 'data/Leg/IMG0001.dcm')
    ENVMAP = os.path.join(ROOT_DIR, 'data/table_mountain_2_puresky_1k.hdr')
    TRANSFER_FUNC = "data/lut_leg/muscle_gen_two.txt"
    OUT_FORMAT = ".txt"

    POINTCLOUD = True # Create initial pointcloud 
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


    cameras = {}
    images = {}
    points3D = {}

    # HACK: write world-space AABB of volume as point3D (pos + rgb) to dataset
    points3D[0] = colmap.Point3D(id=0, xyz=np.array(renderer.volume.AABB("density")[0]), rgb=np.array(renderer.volume.AABB("density")[1]), error=0, image_ids=np.array([]), point2D_idxs=np.array([]))

    focal_original = renderer.colmap_focal_length()
    focal_adjusted = focal_original
    # Adjust focal length for colmap
    focal_adjusted = focal_original * (math.tan(math.radians(FOV_COLMAP/2)) / math.tan(math.radians(FOV_RENDER/2)))

    # write camera
    cameras[0] = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([focal_adjusted, renderer.resolution().x//2, renderer.resolution().y//2]))

    # random sampler
    samplerOut = qmc.Sobol(d=2, seed=SEED+1)
    rng = np.random.default_rng()

    startTime = time.time()
    startTimeAbs = startTime

    cam_positions = []

    img_path = os.path.join(OUT_PATH, "images")
    os.makedirs(img_path, exist_ok=True)



    rs = RenderServer(host='localhost', port=9999)
    rs.start_server()
