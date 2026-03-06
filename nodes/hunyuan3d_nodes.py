import sys
import os
import uuid
import torch
import trimesh
import folder_paths
import numpy as np
from PIL import Image

COMFY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

from comfy_api.latest import Types

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "hy3dpaint"))

# ===== torchvision fix =====
try:
    from torchvision_fix import apply_fix
    apply_fix()
except:
    pass

# =============================
# Load Model
# =============================
_MODEL_CACHE = None


def get_models():
    global _MODEL_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    from hy3dshape import (
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
        Hunyuan3DDiTFlowMatchingPipeline,
    )
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover

    device = "cuda" if torch.cuda.is_available() else "cpu"

    shape_model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1",
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        device=device,
        local_files_only=True
    )

    rmbg = BackgroundRemover()
    floater = FloaterRemover()
    degenerate = DegenerateFaceRemover()
    reducer = FaceReducer()

    try:
        from hy3dpaint.textureGenPipeline import (
            Hunyuan3DPaintPipeline,
            Hunyuan3DPaintConfig
        )

        conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
        conf.realesrgan_ckpt_path = os.path.join(PROJECT_ROOT, "hy3dpaint/ckpt/RealESRGAN_x4plus.pth")
        conf.multiview_cfg_path = os.path.join(PROJECT_ROOT, "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml")
        conf.custom_pipeline = os.path.join(PROJECT_ROOT, "hy3dpaint/hunyuanpaintpbr")

        tex_model = Hunyuan3DPaintPipeline(conf)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

    _MODEL_CACHE = (
        shape_model,
        rmbg,
        floater,
        degenerate,
        reducer,
        tex_model,
        export_to_trimesh,
    )

    return _MODEL_CACHE


# =============================
# Image → Mesh
# =============================

class Hunyuan3DShape:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 15.0}),
                "octree_resolution": ("INT", {"default": 256, "min": 64, "max": 512}),
                "seed": ("INT", {"default": 4202798, "min": 1000000, "max": 9999999}),
                "remove_bg": ("BOOLEAN", {"default": True}),
                "simplify_mesh": ("BOOLEAN", {"default": False}),
                "target_faces": ("INT", {"default": 10000, "min": 1000, "max": 5000000}),
            }
        }

    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("mesh", "mesh_obj_path")
    FUNCTION = "generate"
    CATEGORY = "Hunyuan3D"

    def generate(
            self,
            image,
            steps,
            guidance_scale,
            octree_resolution,
            seed,
            remove_bg,
            simplify_mesh,
            target_faces,
    ):

        (
            shape_model,
            rmbg_worker,
            floater_remove_worker,
            degenerate_face_remove_worker,
            face_reduce_worker,
            _,
            export_to_trimesh,
        ) = get_models()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        image = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        if remove_bg:
            pil_image = rmbg_worker(pil_image.convert("RGB"))

        generator = torch.Generator(device=device).manual_seed(seed)

        outputs = shape_model(
            image=pil_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=8000,
            output_type="mesh",
        )

        mesh = export_to_trimesh(outputs)[0]

        mesh = floater_remove_worker(mesh)
        mesh = degenerate_face_remove_worker(mesh)

        if simplify_mesh:
            mesh = face_reduce_worker(mesh, target_faces)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())
        obj_path = os.path.join(output_dir, f"{unique_id}.obj")
        mesh.export(obj_path)

        return mesh, obj_path


# =============================
# Mesh + Image → Textured Mesh
# =============================

class Hunyuan3DTexture:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MESH", "STRING", "STRING")
    RETURN_NAMES = ("textured_mesh", "textured_glb_path", "textured_obj_path")
    FUNCTION = "paint"
    CATEGORY = "Hunyuan3D"

    def paint(self, mesh, image):
        (
            _,
            _,
            _,
            _,
            _,
            tex_pipeline,
            _,
        ) = get_models()

        if tex_pipeline is None:
            raise RuntimeError("Texture model not available")

        image = image[0].cpu().numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        unique_id = str(uuid.uuid4())

        obj_path = os.path.join(output_dir, f"{unique_id}.obj")
        mesh.export(obj_path)

        textured_obj_path = os.path.join(output_dir, f"{unique_id}_textured.obj")
        textured_glb_path = os.path.join(output_dir, f"{unique_id}_textured.glb")

        print(f"Running texture pipeline... Output: {textured_obj_path}")
        tex_pipeline(
            mesh_path=obj_path,
            image_path=pil_image,
            output_mesh_path=textured_obj_path,
            save_glb=False,
        )

        base_name = os.path.splitext(textured_obj_path)[0]

        textures = {}

        albedo_path = f"{base_name}.jpg"
        metallic_path = f"{base_name}_metallic.jpg"
        roughness_path = f"{base_name}_roughness.jpg"
        normal_path = f"{base_name}_normal.jpg"

        if os.path.exists(albedo_path):
            textures['albedo'] = albedo_path
            print(f"Found albedo: {albedo_path}")
        else:
            albedo_png = f"{base_name}.png"
            if os.path.exists(albedo_png):
                textures['albedo'] = albedo_png
                print(f"Found albedo (png): {albedo_png}")

        if os.path.exists(metallic_path):
            textures['metallic'] = metallic_path
            print(f"Found metallic: {metallic_path}")
        else:
            for alt in [f"{base_name}_Metallic.jpg", f"{base_name}_metallic.png", f"{base_name}_Metallic.png"]:
                if os.path.exists(alt):
                    textures['metallic'] = alt
                    print(f"Found metallic (alt): {alt}")
                    break

        if os.path.exists(roughness_path):
            textures['roughness'] = roughness_path
            print(f"Found roughness: {roughness_path}")
        else:
            for alt in [f"{base_name}_Roughness.jpg", f"{base_name}_roughness.png", f"{base_name}_Roughness.png"]:
                if os.path.exists(alt):
                    textures['roughness'] = alt
                    print(f"Found roughness (alt): {alt}")
                    break

        if os.path.exists(normal_path):
            textures['normal'] = normal_path
            print(f"Found normal: {normal_path}")

        print("-" * 30)
        print(f"Final textures dict passed to converter: {textures}")
        if 'metallic' not in textures or 'roughness' not in textures:
            print("!!! CRITICAL WARNING: Metallic/Roughness map missing! Result will be non-metallic.")
        print("-" * 30)

        if not textures:
            print("No textures found, loading raw OBJ.")
            final_mesh = trimesh.load(textured_obj_path, force='mesh')
        else:
            try:
                from hy3dpaint.convert_utils import create_glb_with_pbr_materials

                valid_textures = {k: v for k, v in textures.items() if os.path.exists(v)}

                print(f"Calling create_glb_with_pbr_materials...")
                create_glb_with_pbr_materials(textured_obj_path, valid_textures, textured_glb_path)

                print(f"GLB generated: {textured_glb_path}")
                final_mesh = trimesh.load(textured_glb_path, force='mesh')
                print("Successfully loaded textured GLB.")

                if hasattr(final_mesh.visual, 'material'):
                    print(f"Material type: {type(final_mesh.visual.material).__name__}")
                    if hasattr(final_mesh.visual.material, 'metallicFactor'):
                        print(f"Metallic Factor: {final_mesh.visual.material.metallicFactor}")
                        print(f"Roughness Factor: {final_mesh.visual.material.roughnessFactor}")

            except ImportError:
                print("Error: create_glb_with_pbr_materials not found.")
                final_mesh = trimesh.load(textured_obj_path, force='mesh')
            except Exception as e:
                print(f"Error during conversion: {e}")
                import traceback
                traceback.print_exc()
                final_mesh = trimesh.load(textured_obj_path, force='mesh')

        return final_mesh, textured_glb_path, textured_obj_path


# =============================
# Save Mesh
# =============================
class SaveMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",),
                "format": (["glb", "obj", "ply", "stl"],),
                "filename_prefix": ("STRING", {"default": "mesh/saved_mesh"})
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "Hunyuan3D"
    OUTPUT_NODE = True

    def save(self, mesh, format, filename_prefix):

        output_dir, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(
                filename_prefix,
                folder_paths.get_output_directory()
            )

        os.makedirs(output_dir, exist_ok=True)

        results = []

        if isinstance(mesh, trimesh.Trimesh):
            file_path = os.path.join(
                output_dir,
                f"{filename}_{counter:05}_.{format}"
            )

            mesh.export(file_path)

            results.append({
                "filename": os.path.basename(file_path),
                "subfolder": subfolder,
                "type": "output"
            })
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh)}")

        return {"ui": {"3d": results}}


# =============================
# Load Mesh
# =============================

class LoadMesh:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING",)
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "load"
    CATEGORY = "Hunyuan3D"

    def load(self, path):
        mesh = trimesh.load(path)
        return (mesh,)


class OutputMeshToComfy:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": ("MESH",)
            }
        }

    RETURN_TYPES = ("MESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "convert"
    CATEGORY = "Hunyuan3D"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def convert(self, mesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            if hasattr(mesh.vertices, "numpy"):
                vertices_np = mesh.vertices.numpy().astype(np.float32)
            else:
                vertices_np = np.array(mesh.vertices, dtype=np.float32)

            if hasattr(mesh.faces, "numpy"):
                faces_np = mesh.faces.numpy().astype(np.int32)
            else:
                faces_np = np.array(mesh.faces, dtype=np.int32)

            # [B, V, 3] 和 [B, F, 3]
            vertices = torch.from_numpy(vertices_np).unsqueeze(0)
            faces = torch.from_numpy(faces_np).unsqueeze(0)

            official_mesh = Types.MESH(vertices, faces)

            return (official_mesh,)
        else:
            raise TypeError(f"Unsupported mesh type: {type(mesh)}. Expected trimesh.Trimesh or similar.")


class ConvertAndSaveMesh:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_mesh": ("*",),
                "output_format": (["obj", "glb", "gltf", "ply", "stl", "off"], {
                    "default": "glb"
                }),
                "filename_prefix": ("STRING", {
                    "default": "Hunyuan3D_Converted",
                })
            },
            "optional": {
                "apply_subdivision": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "是否应用细分平滑 (仅针对某些格式有效)"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "convert_and_save"
    CATEGORY = "Hunyuan3D/Utils"

    def convert_and_save(self, input_mesh, output_format="glb", filename_prefix="Hunyuan3D-2-1_Converted",
                         apply_subdivision=False):

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        loaded_mesh = None

        if isinstance(input_mesh, trimesh.Trimesh) or isinstance(input_mesh, trimesh.Scene):
            loaded_mesh = input_mesh
            print(
                f"[ConvertMesh] Input is trimesh object. Vertices: {len(loaded_mesh.vertices) if hasattr(loaded_mesh, 'vertices') else 'Scene'}")

        elif isinstance(input_mesh, str):
            if not os.path.exists(input_mesh):
                raise FileNotFoundError(f"Input file not found: {input_mesh}")
            print(f"[ConvertMesh] Loading from path: {input_mesh}")
            loaded_mesh = trimesh.load(input_mesh, force='mesh')

        elif isinstance(input_mesh, tuple) and len(input_mesh) == 2:
            verts_tensor, faces_tensor = input_mesh
            v_np = verts_tensor[0].detach().cpu().numpy()
            f_np = faces_tensor[0].detach().cpu().numpy()
            loaded_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
            print(f"[ConvertMesh] Converted from ComfyUI MESH tuple.")

        else:
            if hasattr(input_mesh, "vertices") and hasattr(input_mesh, "faces"):
                if hasattr(input_mesh.vertices, "cpu"):
                    v_np = input_mesh.vertices.cpu().numpy()
                    f_np = input_mesh.faces.cpu().numpy()
                elif hasattr(input_mesh.vertices, "numpy"):
                    v_np = input_mesh.vertices.numpy()
                    f_np = input_mesh.faces.numpy()
                else:
                    v_np = np.array(input_mesh.vertices)
                    f_np = np.array(input_mesh.faces)
                loaded_mesh = trimesh.Trimesh(vertices=v_np, faces=f_np)
                print(f"[ConvertMesh] Converted from generic object with vertices/faces.")
            else:
                raise TypeError(f"Unsupported input type: {type(input_mesh)}. Expected trimesh, path, or tensor tuple.")

        if apply_subdivision and isinstance(loaded_mesh, trimesh.Trimesh):
            try:
                loaded_mesh = loaded_mesh.subdivide()
                print("[ConvertMesh] Subdivision applied.")
            except Exception as e:
                print(f"[ConvertMesh] Subdivision failed: {e}")

        unique_id = str(uuid.uuid4())[:8]
        safe_prefix = "".join([c for c in filename_prefix if c.isalnum() or c in (' ', '_', '-')]).rstrip()
        output_filename = f"{safe_prefix}_{unique_id}.{output_format}"
        output_path = os.path.join(output_dir, output_filename)

        try:
            if isinstance(loaded_mesh, trimesh.Scene):
                loaded_mesh.export(output_path)
            else:
                loaded_mesh.export(output_path)

            print(f"[ConvertMesh] Successfully saved to: {output_path}")

            return (output_path,)

        except Exception as e:
            error_msg = f"Failed to export mesh: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg)

NODE_CLASS_MAPPINGS = {
    "Hunyuan3D21Shape": Hunyuan3DShape,
    "Hunyuan3D21Texture": Hunyuan3DTexture,
    "SaveMesh": SaveMesh,
    "LoadMesh": LoadMesh,
    "OutputMeshToComfy": OutputMeshToComfy,
    "ConvertAndSaveMesh": ConvertAndSaveMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hunyuan3D21Shape": "Hunyuan3D2-1 Shape (Image → Mesh)",
    "Hunyuan3D21Texture": "Hunyuan3D2-1 Texture (Image + Mesh)",
    "SaveMesh": "Save Mesh",
    "LoadMesh": "Load Mesh",
    "OutputMeshToComfy": "Output Mesh To Comfy",
    "ConvertAndSaveMesh": "Convert And Save Mesh",
}
