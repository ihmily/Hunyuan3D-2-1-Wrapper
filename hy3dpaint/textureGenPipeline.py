# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time
import torch
import copy
import trimesh
import numpy as np
from PIL import Image
from typing import List
from DifferentiableRenderer.MeshRender import MeshRender
from .utils.simplify_mesh_utils import remesh_mesh
from .utils.multiview_utils import multiviewDiffusionNet
from .utils.pipeline_utils import ViewProcessor
from .utils.image_super_utils import imageSuperNet
from .utils.uvwrap_utils import mesh_uv_wrap
from DifferentiableRenderer.mesh_utils import convert_obj_to_glb
import warnings

warnings.filterwarnings("ignore")
from diffusers.utils import logging as diffusers_logging

diffusers_logging.set_verbosity(50)


def _log_timing(name, start):
    print(f"[Hunyuan3D][Timing] {name}: {time.perf_counter() - start:.2f}s")


class Hunyuan3DPaintConfig:
    def __init__(self, max_num_view, resolution):
        self.device = "cuda"

        self.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        self.custom_pipeline = "hunyuanpaintpbr"
        self.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        self.dino_ckpt_path = "facebook/dinov2-giant"
        self.realesrgan_ckpt_path = "ckpt/RealESRGAN_x4plus.pth"

        self.raster_mode = "cr"
        self.bake_mode = "back_sample"
        self.render_size = 1024 * 2
        self.texture_size = 1024 * 4
        self.max_selected_view_num = max_num_view
        self.resolution = resolution
        self.bake_exp = 4
        self.merge_method = "fast"
        self.use_super_resolution = True
        self.use_uv_inpaint = True

        # view selection
        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        for azim in range(0, 360, 30):
            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(20)
            self.candidate_view_weights.append(0.01)

            self.candidate_camera_azims.append(azim)
            self.candidate_camera_elevs.append(-20)
            self.candidate_view_weights.append(0.01)


class Hunyuan3DPaintPipeline:

    def __init__(self, config=None) -> None:
        init_start = time.perf_counter()
        self.config = config if config is not None else Hunyuan3DPaintConfig()
        self.models = {}
        self.stats_logs = {}
        render_start = time.perf_counter()
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            bake_mode=self.config.bake_mode,
            raster_mode=self.config.raster_mode,
        )
        _log_timing("texture MeshRender init", render_start)
        self.view_processor = ViewProcessor(self.config, self.render)
        self.load_models()
        _log_timing("texture paint pipeline init total", init_start)

    def load_models(self):
        torch.cuda.empty_cache()
        if getattr(self.config, "use_super_resolution", True):
            self._ensure_super_model()
        multiview_start = time.perf_counter()
        self.models["multiview_model"] = multiviewDiffusionNet(self.config)
        _log_timing("texture multiview model load", multiview_start)
        print("Models Loaded.")

    def _ensure_super_model(self):
        if "super_model" in self.models:
            return
        super_start = time.perf_counter()
        self.models["super_model"] = imageSuperNet(self.config)
        _log_timing("texture RealESRGAN load", super_start)

    @torch.no_grad()
    def __call__(
        self,
        mesh_path=None,
        image_path=None,
        output_mesh_path=None,
        use_remesh=True,
        save_glb=True,
        max_selected_view_num=None,
        resolution=None,
        render_size=None,
        texture_size=None,
        use_super_resolution=None,
        use_uv_inpaint=None,
    ):
        """Generate texture for 3D mesh using multiview diffusion"""
        total_start = time.perf_counter()
        original_max_views = self.config.max_selected_view_num
        original_resolution = self.config.resolution
        original_render_size = self.render.default_resolution
        original_texture_size = self.render.texture_size
        if max_selected_view_num is not None:
            self.config.max_selected_view_num = max(6, int(max_selected_view_num))
        if resolution is not None:
            self.config.resolution = int(resolution)
        if render_size is not None:
            self.config.render_size = int(render_size)
            self.render.set_default_render_resolution(self.config.render_size)
        if texture_size is not None:
            self.config.texture_size = int(texture_size)
            self.render.set_default_texture_resolution(self.config.texture_size)
        if use_super_resolution is None:
            use_super_resolution = getattr(self.config, "use_super_resolution", True)
        if use_uv_inpaint is None:
            use_uv_inpaint = getattr(self.config, "use_uv_inpaint", True)

        # Ensure image_prompt is a list
        try:
            if isinstance(image_path, str):
                image_prompt = Image.open(image_path)
            elif isinstance(image_path, Image.Image):
                image_prompt = image_path
            if not isinstance(image_prompt, List):
                image_prompt = [image_prompt]
            else:
                image_prompt = image_path

            # Process mesh
            path = os.path.dirname(mesh_path)
            if use_remesh:
                step_start = time.perf_counter()
                processed_mesh_path = os.path.join(path, "white_mesh_remesh.obj")
                remesh_mesh(mesh_path, processed_mesh_path)
                _log_timing("texture remesh", step_start)
            else:
                processed_mesh_path = mesh_path

            # Output path
            if output_mesh_path is None:
                output_mesh_path = os.path.join(path, f"textured_mesh.obj")

            # Load mesh
            step_start = time.perf_counter()
            mesh = trimesh.load(processed_mesh_path)
            mesh = mesh_uv_wrap(mesh)
            self.render.load_mesh(mesh=mesh)
            _log_timing("texture load mesh + uv wrap", step_start)

            ########### View Selection #########
            step_start = time.perf_counter()
            selected_camera_elevs, selected_camera_azims, selected_view_weights = self.view_processor.bake_view_selection(
                self.config.candidate_camera_elevs,
                self.config.candidate_camera_azims,
                self.config.candidate_view_weights,
                self.config.max_selected_view_num,
            )
            _log_timing("texture view selection", step_start)

            step_start = time.perf_counter()
            normal_maps = self.view_processor.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, use_abs_coor=True
            )
            position_maps = self.view_processor.render_position_multiview(selected_camera_elevs, selected_camera_azims)
            _log_timing("texture render normal + position maps", step_start)

            ##########  Style  ###########
            image_caption = "high quality"
            image_style = []
            for image in image_prompt:
                image = image.resize((512, 512))
                if image.mode == "RGBA":
                    white_bg = Image.new("RGB", image.size, (255, 255, 255))
                    white_bg.paste(image, mask=image.getchannel("A"))
                    image = white_bg
                image_style.append(image)
            image_style = [image.convert("RGB") for image in image_style]

            ###########  Multiview  ##########
            step_start = time.perf_counter()
            multiviews_pbr = self.models["multiview_model"](
                image_style,
                normal_maps + position_maps,
                prompt=image_caption,
                custom_view_size=self.config.resolution,
                resize_input=True,
            )
            _log_timing("texture multiview PBR generation", step_start)
            ###########  Enhance  ##########
            enhance_images = {}
            enhance_images["albedo"] = copy.deepcopy(multiviews_pbr["albedo"])
            enhance_images["mr"] = copy.deepcopy(multiviews_pbr["mr"])

            if use_super_resolution:
                self._ensure_super_model()
                step_start = time.perf_counter()
                for i in range(len(enhance_images["albedo"])):
                    enhance_images["albedo"][i] = self.models["super_model"](enhance_images["albedo"][i])
                    enhance_images["mr"][i] = self.models["super_model"](enhance_images["mr"][i])
                _log_timing("texture RealESRGAN enhance", step_start)
            else:
                print("[Hunyuan3D][Timing] texture RealESRGAN enhance: skipped")

            ###########  Bake  ##########
            step_start = time.perf_counter()
            for i in range(len(enhance_images)):
                enhance_images["albedo"][i] = enhance_images["albedo"][i].resize(
                    (self.config.render_size, self.config.render_size)
                )
                enhance_images["mr"][i] = enhance_images["mr"][i].resize((self.config.render_size, self.config.render_size))
            _log_timing("texture resize enhanced maps", step_start)
            step_start = time.perf_counter()
            texture, mask = self.view_processor.bake_from_multiview(
                enhance_images["albedo"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            texture_mr, mask_mr = self.view_processor.bake_from_multiview(
                enhance_images["mr"], selected_camera_elevs, selected_camera_azims, selected_view_weights
            )
            mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
            _log_timing("texture bake multiview maps", step_start)

            ##########  inpaint  ###########
            step_start = time.perf_counter()
            if use_uv_inpaint:
                texture = self.view_processor.texture_inpaint(texture, mask_np)
                if "mr" in enhance_images:
                    texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np)
            else:
                texture = self.view_processor.texture_inpaint(texture, mask_np, defualt=[1.0, 1.0, 1.0])
                if "mr" in enhance_images:
                    texture_mr = self.view_processor.texture_inpaint(texture_mr, mask_mr_np, defualt=[0.0, 1.0, 0.0])
            self.render.set_texture(texture, force_set=True)
            if "mr" in enhance_images:
                self.render.set_texture_mr(texture_mr)
            _log_timing("texture inpaint", step_start)

            step_start = time.perf_counter()
            self.render.save_mesh(output_mesh_path, downsample=True)
            _log_timing("texture save obj", step_start)

            if save_glb:
                step_start = time.perf_counter()
                convert_obj_to_glb(output_mesh_path, output_mesh_path.replace(".obj", ".glb"))
                output_glb_path = output_mesh_path.replace(".obj", ".glb")
                _log_timing("texture save glb", step_start)

            _log_timing("texture pipeline total", total_start)
            return output_mesh_path
        finally:
            self.config.max_selected_view_num = original_max_views
            self.config.resolution = original_resolution
            self.config.render_size = original_render_size[0]
            self.config.texture_size = original_texture_size[0]
            self.render.set_default_render_resolution(original_render_size)
            self.render.set_default_texture_resolution(original_texture_size)
