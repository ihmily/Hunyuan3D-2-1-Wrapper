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
import random
import numpy as np
from PIL import Image
from typing import List
import huggingface_hub
from omegaconf import OmegaConf
from diffusers import DiffusionPipeline
from diffusers import EulerAncestralDiscreteScheduler, DDIMScheduler, UniPCMultistepScheduler


def _offline_enabled(config):
    return getattr(config, "local_files_only", False) or os.environ.get("HF_HUB_OFFLINE", "0") == "1"


def _resolve_local_subfolder(model_root, subfolder):
    expanded_root = os.path.abspath(os.path.expanduser(model_root))
    if os.path.isdir(os.path.join(expanded_root, subfolder)):
        return os.path.join(expanded_root, subfolder)
    if os.path.basename(expanded_root) == subfolder and os.path.isdir(expanded_root):
        return expanded_root
    return None


def _log_timing(name, start):
    print(f"[Hunyuan3D][Timing] {name}: {time.perf_counter() - start:.2f}s")


class multiviewDiffusionNet:
    def __init__(self, config) -> None:
        init_start = time.perf_counter()
        self.device = config.device

        cfg_path = config.multiview_cfg_path
        custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")
        cfg = OmegaConf.load(cfg_path)
        self.cfg = cfg
        self.mode = self.cfg.model.params.stable_diffusion_config.custom_pipeline[2:]

        subfolder = "hunyuan3d-paintpbr-v2-1"
        model_path = _resolve_local_subfolder(config.multiview_pretrained_path, subfolder)
        if model_path is None:
            if _offline_enabled(config):
                raise FileNotFoundError(
                    f"Paint model subfolder '{subfolder}' was not found under "
                    f"{config.multiview_pretrained_path}. Offline loading is enabled."
                )
            model_path = huggingface_hub.snapshot_download(
                repo_id=config.multiview_pretrained_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            model_path = os.path.join(model_path, subfolder)

        load_kwargs = dict(
            custom_pipeline=custom_pipeline,
            torch_dtype=torch.float16,
            local_files_only=_offline_enabled(config),
            trust_remote_code=True,
        )
        if getattr(config, "skip_unused_text_components", True):
            load_kwargs.update(
                text_encoder=None,
                tokenizer=None,
                feature_extractor=None,
                safety_checker=None,
                requires_safety_checker=False,
            )

        load_start = time.perf_counter()
        try:
            pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
        except Exception as e:
            if not getattr(config, "skip_unused_text_components", True):
                raise
            print(
                "[Hunyuan3D] Fast texture load without text components failed; "
                f"falling back to full pipeline load. Error: {e}"
            )
            load_kwargs.pop("text_encoder", None)
            load_kwargs.pop("tokenizer", None)
            load_kwargs.pop("feature_extractor", None)
            load_kwargs.pop("safety_checker", None)
            load_kwargs.pop("requires_safety_checker", None)
            pipeline = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
        _log_timing("texture DiffusionPipeline.from_pretrained", load_start)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
        pipeline.set_progress_bar_config(disable=True)
        pipeline.eval()
        setattr(pipeline, "view_size", cfg.model.params.get("view_size", 320))
        to_start = time.perf_counter()
        self.pipeline = pipeline.to(self.device)
        _log_timing("texture pipeline.to(cuda)", to_start)

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            from hunyuanpaintpbr.unet.modules import Dino_v2
            dino_start = time.perf_counter()
            self.dino_v2 = Dino_v2(config.dino_ckpt_path).to(torch.float16)
            self.dino_v2 = self.dino_v2.to(self.device)
            _log_timing("texture Dino_v2 load", dino_start)
        _log_timing("texture multiview model init total", init_start)

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, images, conditions, prompt=None, custom_view_size=None, resize_input=False):
        pils = self.forward_one(
            images, conditions, prompt=prompt, custom_view_size=custom_view_size, resize_input=resize_input
        )
        return pils

    def forward_one(self, input_images, control_images, prompt=None, custom_view_size=None, resize_input=False):
        self.seed_everything(0)
        custom_view_size = custom_view_size if custom_view_size is not None else self.pipeline.view_size
        if not isinstance(input_images, List):
            input_images = [input_images]
        if not resize_input:
            input_images = [
                input_image.resize((self.pipeline.view_size, self.pipeline.view_size)) for input_image in input_images
            ]
        else:
            input_images = [input_image.resize((custom_view_size, custom_view_size)) for input_image in input_images]
        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((custom_view_size, custom_view_size))
            if control_images[i].mode == "L":
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode="1")
        kwargs = dict(generator=torch.Generator(device=self.pipeline.device).manual_seed(0))

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        kwargs["width"] = custom_view_size
        kwargs["height"] = custom_view_size
        kwargs["num_in_batch"] = num_view
        kwargs["images_normal"] = normal_image
        kwargs["images_position"] = position_image

        if hasattr(self.pipeline.unet, "use_dino") and self.pipeline.unet.use_dino:
            dino_start = time.perf_counter()
            dino_hidden_states = self.dino_v2(input_images[0])
            kwargs["dino_hidden_states"] = dino_hidden_states
            _log_timing("texture DINO encode", dino_start)

        sync_condition = None

        infer_steps_dict = {
            "EulerAncestralDiscreteScheduler": 30,
            "UniPCMultistepScheduler": 15,
            "DDIMScheduler": 50,
            "ShiftSNRScheduler": 15,
        }

        diffusion_start = time.perf_counter()
        mvd_image = self.pipeline(
            input_images[0:1],
            num_inference_steps=infer_steps_dict[self.pipeline.scheduler.__class__.__name__],
            prompt=prompt,
            sync_condition=sync_condition,
            guidance_scale=3.0,
            **kwargs,
        ).images
        _log_timing("texture multiview diffusion", diffusion_start)

        if "pbr" in self.mode:
            mvd_image = {"albedo": mvd_image[:num_view], "mr": mvd_image[num_view:]}
            # mvd_image = {'albedo':mvd_image[:num_view]}
        else:
            mvd_image = {"hdr": mvd_image}

        return mvd_image
