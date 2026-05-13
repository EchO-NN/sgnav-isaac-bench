from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from isaac_bench.config import str_to_bool


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)], dtype=np.float32)


class IsaacSimServer:
    def __init__(
        self,
        headless: bool = True,
        width: int = 640,
        height: int = 480,
        verbose: bool = True,
        camera_hfov_deg: float = 110.0,
        mast_height_m: float = 1.35,
        forward_offset_m: float = 0.0,
        camera_pitch_deg: float = 0.0,
        camera_near_m: float = 0.02,
        camera_far_m: float = 80.0,
        enable_depth: bool = False,
        camera_annotator_device: str = "cuda",
    ):
        self.headless = bool(headless)
        self.width = int(width)
        self.height = int(height)
        self.verbose = bool(verbose)
        self.camera_hfov_deg = float(camera_hfov_deg)
        self.mast_height_m = float(mast_height_m)
        self.forward_offset_m = float(forward_offset_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.camera_near_m = float(camera_near_m)
        self.camera_far_m = float(camera_far_m)
        self.enable_depth = bool(enable_depth)
        device = str(camera_annotator_device or "cpu").strip().lower()
        self.camera_annotator_device = device if device in {"cpu", "cuda"} else "cpu"
        self.app = None
        self.world = None
        self.robot = None
        self.controller = None
        self.camera = None
        self.camera_prim_path = "/World/Kaya/camera_rgbd"
        self.kinematic_pose: Optional[Tuple[float, float, float, float]] = None
        self.last_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.last_rgb_gpu = None
        self.last_rgb_device = "cpu"
        self.last_depth = np.zeros((self.height, self.width), dtype=np.float32)
        self._logged_cuda_rgb_fallback = False

    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def start(self) -> None:
        from isaacsim import SimulationApp

        self.log("[isaac] starting SimulationApp")
        self.app = SimulationApp({"headless": self.headless})
        self.log("[isaac] SimulationApp ready")

    def load_scene(self, usd_path: str) -> None:
        if self.app is None:
            self.start()
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import is_stage_loading, open_stage

        self.log("[isaac] opening stage %s" % usd_path)
        open_stage(str(usd_path))
        while is_stage_loading():
            self.app.update()
        self.disable_imported_scene_rigid_bodies()
        self.world = World(stage_units_in_meters=1.0)
        self.log("[isaac] stage loaded")

    def disable_imported_scene_rigid_bodies(self) -> None:
        try:
            import omni.usd
            from pxr import UsdPhysics
        except Exception as exc:
            self.log("[isaac] scene rigid-body cleanup skipped: %s" % exc)
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        removed = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/Root/Meshes"):
                continue
            try:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    removed += 1
            except Exception:
                continue
        if removed:
            self.log("[isaac] disabled %d imported scene rigid bodies" % removed)

    def spawn_kaya(self, pose_world: Tuple[float, float, float, float]) -> None:
        from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
        from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
        from isaacsim.storage.native import get_assets_root_path

        from isaac_bench.robot.kaya_spawn import spawn_kaya

        self.log("[isaac] spawning Kaya")
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac assets root is unavailable; cannot find Kaya USD")
        kaya_asset_path = assets_root + "/Isaac/Robots/NVIDIA/Kaya/kaya.usd"
        x, y, z, yaw = pose_world
        self.robot = spawn_kaya(
            self.world,
            "/World/Kaya",
            "my_kaya",
            kaya_asset_path,
            np.asarray([x, y, z], dtype=np.float32),
            yaw_to_quat_wxyz(yaw),
        )
        kaya_setup = HolonomicRobotUsdSetup(robot_prim_path=self.robot.prim_path, com_prim_path="/World/Kaya/base_link/control_offset")
        wheel_radius, wheel_positions, wheel_orientations, mecanum_angles, wheel_axis, up_axis = kaya_setup.get_holonomic_controller_params()
        self.controller = HolonomicController(
            name="holonomic_controller",
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
            wheel_axis=wheel_axis,
            up_axis=up_axis,
        )

    def camera_pose_from_base(self, pose_world: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x, y, z, yaw = [float(v) for v in pose_world]
        cam_x = x + math.cos(yaw) * self.forward_offset_m
        cam_y = y + math.sin(yaw) * self.forward_offset_m
        cam_z = z + self.mast_height_m
        return cam_x, cam_y, cam_z, yaw

    def camera_orientation(self, yaw: float):
        from isaacsim.core.utils import rotations as rot_utils

        return rot_utils.euler_angles_to_quat(
            np.asarray([self.camera_pitch_deg, 0.0, math.degrees(float(yaw))], dtype=np.float32),
            degrees=True,
        )

    def configure_camera_intrinsics(self) -> None:
        if self.camera is None:
            return
        hfov = min(max(float(self.camera_hfov_deg), 30.0), 150.0)
        aperture = 20.955
        focal_length = aperture / (2.0 * math.tan(math.radians(hfov) * 0.5))
        try:
            self.camera.set_horizontal_aperture(aperture)
            self.camera.set_focal_length(focal_length)
            self.log("[isaac] camera hfov %.1f deg, focal %.2f" % (hfov, focal_length))
        except Exception as exc:
            self.log("[isaac] camera intrinsics fallback: %s" % exc)
        self.configure_camera_clipping()

    def configure_camera_clipping(self) -> None:
        if self.camera is None:
            return
        near = max(0.001, float(self.camera_near_m))
        far = max(near + 0.1, float(self.camera_far_m))
        applied = False
        try:
            self.camera.set_clipping_range(near, far)
            applied = True
        except Exception:
            pass
        try:
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.camera_prim_path) if stage is not None else None
            if prim is not None and prim.IsValid():
                UsdGeom.Camera(prim).GetClippingRangeAttr().Set(Gf.Vec2f(near, far))
                applied = True
        except Exception as exc:
            if not applied:
                self.log("[isaac] camera clipping fallback: %s" % exc)
        if applied:
            self.log("[isaac] camera clipping %.3f..%.1f m" % (near, far))

    def attach_camera(self, pose_world: Tuple[float, float, float, float]) -> None:
        from isaacsim.sensors.camera import Camera

        self.log("[isaac] attaching %s camera" % ("RGB-D" if self.enable_depth else "RGB"))
        self.log("[isaac] camera annotator device %s" % self.camera_annotator_device)
        cam_x, cam_y, cam_z, yaw = self.camera_pose_from_base(pose_world)
        camera_kwargs = {
            "prim_path": self.camera_prim_path,
            "position": np.asarray([cam_x, cam_y, cam_z], dtype=np.float32),
            "frequency": 20,
            "resolution": (self.width, self.height),
            "orientation": self.camera_orientation(yaw),
            "annotator_device": self.camera_annotator_device,
        }
        try:
            self.camera = Camera(**camera_kwargs)
        except TypeError:
            camera_kwargs.pop("annotator_device", None)
            self.camera = Camera(**camera_kwargs)
        self.configure_camera_intrinsics()
        self.bind_viewport_to_robot_camera()

    def bind_viewport_to_robot_camera(self) -> None:
        if self.headless:
            return
        try:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.camera_path = self.camera_prim_path
                self.log("[isaac] viewport camera -> %s" % self.camera_prim_path)
                return
        except Exception:
            pass
        try:
            from omni.kit.viewport.utility import get_active_viewport_window

            viewport_window = get_active_viewport_window()
            if viewport_window is not None:
                viewport_window.set_active_camera(self.camera_prim_path)
                self.log("[isaac] viewport camera -> %s" % self.camera_prim_path)
        except Exception as exc:
            self.log("[isaac] viewport camera binding skipped: %s" % exc)

    def set_pose_world(self, pose_world: Tuple[float, float, float, float]) -> None:
        x, y, z, yaw = [float(v) for v in pose_world]
        self.kinematic_pose = (x, y, z, yaw)
        quat = yaw_to_quat_wxyz(yaw)
        if self.robot is not None:
            try:
                self.robot.set_world_pose(
                    position=np.asarray([x, y, z], dtype=np.float32),
                    orientation=quat,
                )
            except Exception:
                pass
        if self.camera is not None:
            cam_x, cam_y, cam_z, _ = self.camera_pose_from_base((x, y, z, yaw))
            try:
                self.camera.set_world_pose(
                    position=np.asarray([cam_x, cam_y, cam_z], dtype=np.float32),
                    orientation=self.camera_orientation(yaw),
                )
            except Exception:
                pass

    def reset_episode(
        self,
        usd_path: str,
        pose_world: Tuple[float, float, float, float],
        read_rgb: bool = True,
        rgb_device: Optional[str] = None,
    ) -> dict:
        self.load_scene(usd_path)
        self.spawn_kaya(pose_world)
        self.attach_camera(pose_world)
        self.log("[isaac] resetting world")
        self.world.reset()
        self.set_pose_world(pose_world)
        self.log("[isaac] initializing camera")
        self.camera.initialize()
        self.configure_camera_intrinsics()
        self.bind_viewport_to_robot_camera()
        if self.enable_depth:
            try:
                self.camera.add_distance_to_camera_to_frame()
                self.camera.add_distance_to_image_plane_to_frame()
            except Exception as exc:
                self.log("[isaac] camera annotator fallback: %s" % exc)
        self.log("[isaac] rendering warmup frames")
        for _ in range(60):
            self.app.update()
        return self.get_observation(read_rgb=read_rgb, read_depth=self.enable_depth, rgb_device=rgb_device)

    def step_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        frames: int = 3,
        read_rgb: bool = True,
        read_depth: Optional[bool] = None,
        rgb_device: Optional[str] = None,
    ) -> dict:
        if self.robot is not None and self.controller is not None:
            self.robot.apply_wheel_actions(self.controller.forward(command=[float(vx), float(vy), float(wz)]))
        for _ in range(int(frames)):
            self.world.step(render=True)
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=self.enable_depth if read_depth is None else bool(read_depth),
            rgb_device=rgb_device,
        )

    def step_kinematic_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        dt: float = 0.2,
        render_updates: int = 2,
        read_rgb: bool = True,
        read_depth: Optional[bool] = None,
        rgb_device: Optional[str] = None,
    ) -> dict:
        x, y, z, yaw = self.get_pose_world()
        dx = math.cos(yaw) * float(vx) - math.sin(yaw) * float(vy)
        dy = math.sin(yaw) * float(vx) + math.cos(yaw) * float(vy)
        yaw = yaw + float(wz) * float(dt)
        while yaw > math.pi:
            yaw -= 2.0 * math.pi
        while yaw < -math.pi:
            yaw += 2.0 * math.pi
        self.set_pose_world((x + dx * float(dt), y + dy * float(dt), z, yaw))
        for _ in range(int(render_updates)):
            self.app.update()
        return self.get_observation(
            read_rgb=read_rgb,
            read_depth=self.enable_depth if read_depth is None else bool(read_depth),
            rgb_device=rgb_device,
        )

    def get_pose_world(self) -> Tuple[float, float, float, float]:
        if self.kinematic_pose is not None:
            return self.kinematic_pose
        if self.robot is None:
            return 0.0, 0.0, 0.0, 0.0
        pos, quat = self.robot.get_world_pose()
        # quat is wxyz; yaw only.
        w, _x, _y, z = [float(v) for v in quat]
        yaw = math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)
        return float(pos[0]), float(pos[1]), float(pos[2]), yaw

    def get_observation(self, read_rgb: bool = True, read_depth: bool = False, rgb_device: Optional[str] = None) -> dict:
        rgb = self.last_rgb
        rgb_gpu = self.last_rgb_gpu
        depth = self.last_depth
        frame = {}
        requested_rgb_device = str(rgb_device or self.camera_annotator_device).strip().lower()
        if requested_rgb_device not in {"cpu", "cuda"}:
            requested_rgb_device = "cpu"
        if self.camera is not None:
            if read_depth and self.enable_depth:
                frame = self.camera.get_current_frame() or {}
            if read_rgb:
                if requested_rgb_device == "cuda":
                    try:
                        rgb_gpu = self.camera.get_rgb(device="cuda")
                        self.last_rgb_gpu = rgb_gpu
                        self.last_rgb_device = "cuda"
                    except Exception as exc:
                        if not self._logged_cuda_rgb_fallback:
                            self.log("[isaac] CUDA RGB read failed; falling back to CPU: %s" % exc)
                            self._logged_cuda_rgb_fallback = True
                        requested_rgb_device = "cpu"
                if requested_rgb_device == "cpu":
                    try:
                        rgb_frame = np.asarray(self.camera.get_rgb(device="cpu"))
                    except Exception:
                        rgb_frame = np.asarray(self.camera.get_rgba())
                    if rgb_frame.ndim == 3 and rgb_frame.shape[2] >= 3:
                        rgb = rgb_frame[:, :, :3].astype(np.uint8)
                        self.last_rgb = rgb
                        self.last_rgb_device = "cpu"
                elif isinstance(frame.get("rgb"), np.ndarray):
                    rgb_frame = np.asarray(frame["rgb"])
                    if rgb_frame.ndim == 3 and rgb_frame.shape[2] >= 3:
                        rgb = rgb_frame[:, :, :3].astype(np.uint8)
                        self.last_rgb = rgb
            if read_depth and self.enable_depth:
                depth_value = frame.get("distance_to_camera")
                if not isinstance(depth_value, np.ndarray):
                    depth_value = frame.get("distance_to_image_plane")
                if not isinstance(depth_value, np.ndarray):
                    try:
                        depth_value = self.camera.get_depth(device="cpu")
                    except Exception:
                        depth_value = None
                if isinstance(depth_value, np.ndarray):
                    depth_frame = np.asarray(depth_value, dtype=np.float32)
                    if depth_frame.ndim == 2:
                        depth = depth_frame
                        self.last_depth = depth
        pose = self.get_pose_world()
        return {
            "rgb": rgb,
            "rgb_gpu": rgb_gpu,
            "rgb_device": self.last_rgb_device,
            "depth": depth,
            "has_rgb": bool(read_rgb),
            "has_depth": bool(read_depth and self.enable_depth),
            "pose_world": pose,
            "camera_pose_world": self.camera_pose_from_base(pose),
            "sim_time": 0.0,
            "collided": False,
        }

    def close(self) -> None:
        if self.app is not None:
            self.app.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-usd", required=True)
    parser.add_argument("--spawn", nargs=4, type=float, default=[0.0, 0.0, 0.05, 0.0])
    parser.add_argument("--headless", nargs="?", const=True, default=True, type=str_to_bool)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-frame", default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-hfov-deg", type=float, default=110.0)
    parser.add_argument("--camera-mast-height-m", type=float, default=1.35)
    parser.add_argument("--camera-forward-offset-m", type=float, default=0.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--camera-near-m", type=float, default=0.02)
    parser.add_argument("--camera-far-m", type=float, default=80.0)
    parser.add_argument("--camera-annotator-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--read-depth", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)
    server = IsaacSimServer(
        headless=args.headless,
        verbose=args.verbose,
        camera_hfov_deg=args.camera_hfov_deg,
        mast_height_m=args.camera_mast_height_m,
        forward_offset_m=args.camera_forward_offset_m,
        camera_pitch_deg=args.camera_pitch_deg,
        camera_near_m=args.camera_near_m,
        camera_far_m=args.camera_far_m,
        enable_depth=args.read_depth,
        camera_annotator_device=args.camera_annotator_device,
    )
    try:
        print("[isaac] reset episode", flush=True)
        obs = server.reset_episode(args.scene_usd, tuple(args.spawn), rgb_device="cpu" if args.save_frame else None)
        print("[isaac] observation captured", flush=True)
        if args.save_frame:
            from PIL import Image

            out = Path(args.save_frame)
            out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(obs["rgb"]).save(out)
            depth = obs["depth"]
            finite = np.isfinite(depth)
            if np.any(finite):
                depth_clean = np.where(finite, depth, 0.0)
                depth_max = max(float(np.max(depth_clean)), 1e-6)
            else:
                depth_clean = np.zeros_like(depth, dtype=np.float32)
                depth_max = 1.0
            depth_img = np.clip(depth_clean / depth_max * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(depth_img).save(out.with_name(out.stem + "_depth.png"))
        print({"pose_world": obs["pose_world"], "camera_pose_world": obs["camera_pose_world"]}, flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
