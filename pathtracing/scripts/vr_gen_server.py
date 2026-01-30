import os
import socket
import sys
import time
import math
import asyncio
import struct
import msgpack
import numpy as np
from pathlib import Path
from typing import Dict, Any
from collections import deque
import traceback

# import colmap import/export script
sys.path.append(os.path.dirname(__file__))
import read_write_model as colmap


# import renderer module
import volpy

# import colmap import/export script and shared settings
sys.path.append(str(Path(__file__).parent))
import read_write_model as colmap
# --- MODIFIED: Import the new configuration structure ---
from vr_gen_config import general_settings, server_config, datasets


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

    if config.get("SCALE_ADJUST_Y") is not None:
        renderer.volume.transform.set_value(1, 1, config["SCALE_ADJUST_Y"])
    if config.get("SCALE_ADJUST_Z") is not None:
        renderer.volume.transform.set_value(2, 2, config["SCALE_ADJUST_Z"])

    renderer.seed = config.get("SEED", 42)
    renderer.bounces = config.get("BOUNCES", 4)
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
    
    renderer.cam_fov = config.get("FOV_RENDER", 20)
    
    print("Renderer initialized successfully with the selected configuration.")
    return renderer

class RenderServer:
    def __init__(self, config: Dict[str, Any], renderer: volpy.Renderer):
        self.host = config.get("HOST", "0.0.0.0")
        self.port = config.get("PORT", 9999)
        self.config = config
        self.renderer = renderer
        self.client = None

        self.pending_request = None
        self.request_lock = asyncio.Lock()
        self.waiting_event = asyncio.Event()
        self.render_task = None
        
        self.buffer_in_memory = self.config.get("STORE_RENDER_BUFFER", False)
        self.max_buffer_size = self.config.get("MAX_BUFFER_SIZE", 100)
        
        self.rejection_sampling = self.config.get("REJECTION_SAMPLING_ENABLED", False)
        self.dist_threshold_sq = self.config.get("REJECTION_DISTANCE_THRESHOLD", 0.05) ** 2
        self.angle_threshold = self.config.get("REJECTION_ANGLE_THRESHOLD", 15.0)

        self.render_buffer = None
        self.historical_poses = None
        if self.buffer_in_memory:
            self.render_buffer = deque(maxlen=self.max_buffer_size)
            self.buffer_frame_count = 0
            self.buffer_colmap_camera = {}
            self.buffer_colmap_camera[0] = colmap.Camera(id=0, model="SIMPLE_PINHOLE", width=renderer.resolution().x, height=renderer.resolution().y, params=np.array([renderer.colmap_focal_length(), renderer.resolution().x//2, renderer.resolution().y//2]))
            print(f"In-memory render buffer enabled with a max size of {self.max_buffer_size} frames.")
            if self.rejection_sampling:
                # A parallel buffer just for raw pose vectors for efficient comparison
                self.historical_poses = deque(maxlen=300)
                print(f"Rejection sampling enabled: dist² > {self.dist_threshold_sq}, angle > {self.angle_threshold}°")

    async def _process_buffer(self):
        """
        A placeholder task that demonstrates how the buffer could be processed.
        This is the extension point for sending data to another service.
        """
        from PIL import Image
        while True:
            await asyncio.sleep(5.0) 
            
            # For testing waiting until we have gathered sufficient amount. Could instead directly stream out
            if self.render_buffer and len(self.render_buffer) > 3:
                path = "colmap_output_" + str(time.time()) + "/"
                os.makedirs(path, exist_ok=True)
                images = {}
                for i, ele in enumerate(self.render_buffer):
                    images[i] = ele["pose"]
                    img = Image.frombytes('RGBA', (self.renderer.resolution().x, self.renderer.resolution().y), ele["rgba_bytes"])
                    img.save(f"{path}{images[i].name}")

                colmap.write_model(self.buffer_colmap_camera, images, {}, path, ext=".txt")
                self.render_buffer.clear()
                

    async def start_server(self):
        """Start the server and render worker"""
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        self.render_task = asyncio.create_task(self._event_loop())
        if self.buffer_in_memory:
            asyncio.create_task(self._process_buffer())

        print(f"Render server listening on {self.host}:{self.port}")
        
        async with server:
            await server.serve_forever()

    
    async def handle_client(self, reader, writer):
        """Handle client connection"""

        addr = writer.get_extra_info('peername')
        if self.client is not None:
            print(f"Another client is already connected. Rejecting connection from {addr}")
            writer.close(); await writer.wait_closed()
            return
        
        sock = writer.get_extra_info('socket')
        if sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print(f"Connection from {addr}")
        self.client = writer
        
        # Start the render loop
        
        try:
            while True:
                # Read message length
                length_data = await reader.readexactly(4)
                length = struct.unpack('!I', length_data)[0]
                
                # Read message data
                message_data = await reader.readexactly(length)
                # Apparently quickack is not sticky and should be set each receive?
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1) 

                # Process pose message (non-blocking)
                await self._handle_request(message_data)
                
        except (asyncio.IncompleteReadError, ConnectionResetError):
            print(f"Client {addr} disconnected")
        except Exception as e:
            print(f"Error handling client {addr}: {e}")
        finally:
            # Cancel render task
            
            self.client = None
            self.pending_request = None
            writer.close()
            await writer.wait_closed()
    

    async def _handle_request(self, message_data):
        """Handle message and queue for rendering (non-blocking)"""

        try:
            request = msgpack.unpackb(message_data)
            
            async with self.request_lock:
                self.pending_request = request
                self.waiting_event.set()
                
        except Exception as e:
            print(f"Error handling request: {e}")
    
    async def _event_loop(self):
        """Independent render loop that processes pending requests"""
        
        while True:
            await self.waiting_event.wait()
            async with self.request_lock:
                if self.pending_request:
                    current_request = self.pending_request
                    self.pending_request = None
                else: continue

            try:
                req_type = current_request["type"]
                match req_type:
                    case "frame":
                        await self._handle_render(self.client, current_request)
                    case "ping":
                        await self._handle_ping(self.client, current_request)
                    case "quality":
                        await self._handle_quality(self.client, current_request)
                    case "transferfunction":
                        await self._handle_transfer_function(self.client, current_request)
                    case _:
                        print(f"Unknown request type: {req_type}")
                
            except Exception as e:
                print(f"Error in render loop: {e}")
                traceback.print_exc()

            self.waiting_event.clear()


    async def _handle_ping(self, writer, request):
        """Handle ping request and send response."""
        if not writer:
            print("No client connected, cannot send ping response")
            return
        
        await self._send(writer, "pong", {"time": request["time"]})


    async def _handle_quality(self, writer, request):
        """Handle quality adjustment request."""
        new_samples = request.get("samples")
        new_bounces = request.get("bounces")

        if new_samples is not None:
            self.config["SAMPLES"] = new_samples
            print(f"Updated samples per pixel to {new_samples}")

        if new_bounces is not None:
            self.renderer.bounces = new_bounces
            print(f"Updated bounces to {new_bounces}")

        await self._send(writer, "quality_ack", {"samples": new_samples, "bounces": new_bounces})


    async def _handle_transfer_function(self, writer, request):
        """Handle transfer function update request."""
        root = Path(__file__).resolve().parent.parent
        tf_path = Path(root, request.get("tf_path"))
        if os.path.isfile(tf_path):
            tf = volpy.TransferFunction(str(tf_path))
            tf.window_left = request.get("window_left", 0.0)
            tf.window_width = request.get("window_width", 1.0)
            self.renderer.transferfunc = tf
            print(f"Updated transfer function to {tf_path}")
            await self._send(writer, "tf_ack", {"status": True, "message": "Reloaded TF"})
        else:
            print(f"Invalid transfer function path: {tf_path}")
            await self._send(writer, "tf_ack", {"status": False, "message": "Invalid TF path"})

    async def _handle_render(self, writer, request):
        """Render frame, send to client, and save to buffer if enabled."""

        start_time = time.time()
        
        # Render the frame
        frame_data = await self._render_frame(request["cam"], request["time"])
        
        render_time = time.time() - start_time
        print(f"Render {request['time']} completed in {render_time:.4f}s")
        
        if self.buffer_in_memory:
            self.add_to_buffer(frame_data, request["cam"])

        payload = {
            'time': request["time"],
            'l_rgba': frame_data["l_rgba"],
            'l_depth': frame_data.get("l_depth"),
            'r_rgba': frame_data.get("r_rgba"),
            'r_depth': frame_data.get("r_depth"),
        }

        await self._send(writer, request["type"], payload)


    async def _render_frame(self, pose, timing):
        """Renders a single frame and returns data and the original pose dictionary."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def _render_callback():
            self.renderer.cam_pos = volpy.vec3(*pose["pos"])
            self.renderer.cam_dir = volpy.vec3(*pose["dir"])
            self.renderer.cam_up = volpy.vec3(*pose["up"])

            start_time = time.time()

            samples = self.config.get("SAMPLES", 8)
            self.renderer.render_multi(samples)

            #print(f"Internal render done in {time.time() - start_time:.4f}s")

            self.renderer.process()

            #print(f"Process on top done in {time.time() - start_time:.4f}s")

            left = np.array(self.renderer.rgba_data(), dtype=np.uint8)
            depth_l = np.array(self.renderer.depth_data(), dtype=np.float32)

            #print(f"Data access done in {time.time() - start_time:.4f}s")

            right = None
            depth_r = None

            if self.config.get("GEN_VR", False):
                eye_sep = self.config.get("VR_EYE_SEPARATION", 0.06)
                self.renderer.rightward(eye_sep)

                self.renderer.render_multi(samples)
                self.renderer.process()

                right = np.array(self.renderer.rgba_data(), dtype=np.uint8)
                depth_r = np.array(self.renderer.depth_data(), dtype=np.float32)

            future.set_result((left, depth_l, right, depth_r))

        loop.call_soon_threadsafe(_render_callback)
        rgba_l, depth_l, rgba_r, depth_r = await future

        start_time_outer = time.time()

        # This is not super lightweight i think

        rgba_l = rgba_l.reshape((self.renderer.resolution().y, self.renderer.resolution().x, 4))
        rgba_l = np.flip(rgba_l, axis=0)
        rgba_l_bytes = rgba_l.tobytes()
        frame_data = {"l_rgba": rgba_l_bytes}

        if depth_l is not None:
            depth_l = depth_l.reshape((self.renderer.resolution().y, self.renderer.resolution().x, 1))
            depth_l = np.flip(depth_l, axis=0)
            depth_l_bytes = depth_l.tobytes()
            frame_data["l_depth"] = depth_l_bytes

        if rgba_r is not None:
            rgba_r = rgba_r.reshape((self.renderer.resolution().y, self.renderer.resolution().x, 4))
            rgba_r = np.flip(rgba_r, axis=0)
            rgba_r_bytes = rgba_r.tobytes()
            frame_data["r_rgba"] = rgba_r_bytes
            
            if depth_r is not None:
                depth_r = depth_r.reshape((self.renderer.resolution().y, self.renderer.resolution().x, 1))
                depth_r = np.flip(depth_r, axis=0)
                depth_r_bytes = depth_r.tobytes()
                frame_data["r_depth"] = depth_r_bytes


        #print(f"Outer processing on top done in {time.time() - start_time_outer:.4f}s")

        return frame_data

    def _is_pose_too_similar(self, new_pos: list[float], new_dir: list[float]) -> bool:
        """Checks a new pose against all historical poses in the buffer."""
        if not self.historical_poses:
            return False

        # Assuming the direction vector is already normalized
        np_new_dir = np.array(new_dir)

        for old_pos, old_dir in self.historical_poses:
            
            # Check squared distanc as were just comparing
            dist_sq = (new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2 + (new_pos[2] - old_pos[2])**2
            
            if dist_sq < self.dist_threshold_sq:
                dot_product = np.dot(np_new_dir, old_dir)
                angle = math.degrees(math.acos(np.clip(dot_product, -1.0, 1.0)))
                
                if angle < self.angle_threshold:
                    return True
        
        return False

    def add_to_buffer(self, frame_data: Dict[str, np.ndarray], pose: Dict[str, list[float]]):
        """Adds frame to buffer, performing rejection sampling if enabled."""
        try:
            new_pos = pose["pos"]
            new_dir = pose["dir"]

            if self.rejection_sampling and self._is_pose_too_similar(new_pos, new_dir):
                return 
            
            frame_id = self.buffer_frame_count
            qvec = np.array(self.renderer.colmap_view_rot())[[3, 0, 1, 2]]
            tvec = np.array(self.renderer.colmap_view_trans())
            pose_data = colmap.Image(id=frame_id, qvec=qvec, tvec=tvec, camera_id=0, name=f"frame_{frame_id:06d}.png", xys=[], point3D_ids=[])

            self.render_buffer.append({
                'frame_id': frame_id,
                'rgba_bytes': frame_data["l_rgba"],
                'pose': pose_data,
            })

            if self.rejection_sampling:
                self.historical_poses.append((new_pos, new_dir))

            self.buffer_frame_count += 1
        except Exception as e:
            print(f"Error adding to buffer: {e}")
            
    async def _send(self, writer, type, payload):
        """Send rendered frame back to client."""
        try:
            msg = {'type': type, 'payload': payload}
            response = msgpack.packb(msg)

            length = len(response)
            message = struct.pack('!I', length) + response
            
            writer.write(message)
            await writer.drain()
        except Exception as e:
            print(f"Error sending frame: {e}")


if __name__ == "__main__":
    try:
        dataset_name = server_config.get("DATASET_TO_USE")
        if not dataset_name or dataset_name not in datasets:
            raise ValueError(f"'{dataset_name}' defined in server_config is not a valid dataset.")
        print(f"Server starting with dataset: '{dataset_name}'")
        selected_dataset_config = datasets[dataset_name]
        
        # Create the final config using the cascade: general -> server -> dataset
        final_config = {**general_settings, **server_config, **selected_dataset_config}
        
        renderer_instance = setup_renderer(final_config)
        rs = RenderServer(config=final_config, renderer=renderer_instance)
        asyncio.run(rs.start_server())
    except Exception as e:
        print(f"Failed to start server: {e}")