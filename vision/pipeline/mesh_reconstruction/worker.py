"""MV-SAM3D on the mesh views, with the D405 depth in place of a depth model (sam3d env).

create_mesh.py runs this from the MV-SAM3D checkout with it on PYTHONPATH, so its fork of
sam3d_objects is the one imported:

    python <this file> VIEWS_DIR OUT_DIR --seed 42 --stage1-steps 25 --stage2-steps 25 --tile 320

VIEWS_DIR holds view_<i>/ run folders (rgb.png, depth.npy, camera_intrinsics.json, obj1/mask.png);
view_0 is MV-SAM3D's reference view. Weighting is MV-SAM3D's default (attention entropy in both
stages). SAM 3D's own mesh postprocess (simplify, fill holes) and texture baking (PyTorch3D) are on.
Writes OUT_DIR/mesh.glb (textured, metres, centred in its oriented bounding box so x, y, z run
along its extents) and OUT_DIR/preview.png (input crops, renders with labelled axes, dimensions).
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # as notebook/inference.py
sys.path.append("notebook")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from inference import Inference  # noqa: E402
from pytorch3d.renderer import (BlendParams, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer,  # noqa: E402
                                MeshRenderer, PointLights, RasterizationSettings, TexturesUV,
                                look_at_view_transform)
from pytorch3d.structures import Meshes  # noqa: E402
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils  # noqa: E402
from sam3d_objects.utils.latent_weighting import WeightingConfig  # noqa: E402

# Texture baking renders the splat with the "inria" rasterizer unless told otherwise, and that is
# not installed for the 5090; gsplat is, and the renderer supports it.
_render_frames = render_utils.render_frames
render_utils.render_frames = lambda sample, extrinsics, intrinsics, options={}, **kw: _render_frames(
    sample, extrinsics, intrinsics, {"backend": "gsplat", **options}, **kw)

AXIS_COLORS = ((230, 40, 40), (40, 170, 40), (40, 80, 230))  # x, y, z in RGB


def load_view(view: Path, edge_px: int):
    """(rgb, mask, pointmap). Depth is dropped within edge_px inside the mask's edge, where the mask
    bleeds onto the background and the D405 has flying pixels; the mask itself is unchanged."""
    rgb = cv2.cvtColor(cv2.imread(str(view / "rgb.png")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(view / "obj1/mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    depth = np.load(view / "depth.npy").astype(np.float32)
    K = np.array(json.loads((view / "camera_intrinsics.json").read_text())["K"])
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    z = np.where(depth > 0, depth, np.nan)
    # OpenCV camera frame, (3, H, W): the fork applies the PyTorch3D axis flip itself.
    pointmap = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]).astype(np.float32)
    keep = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=edge_px) > 0
    pointmap[:, mask & ~keep] = np.nan
    return rgb, mask, pointmap


def reconstruct(views, args) -> trimesh.Trimesh:
    inference = Inference("checkpoints/hf/pipeline.yaml", compile=False)
    images, masks, pointmaps = zip(*views)
    result = inference._pipeline.run_multi_view(
        view_images=list(images), view_masks=list(masks), view_pointmaps=list(pointmaps), seed=args.seed,
        mode="multidiffusion", stage1_inference_steps=args.stage1_steps, stage2_inference_steps=args.stage2_steps,
        with_mesh_postprocess=True, with_texture_baking=True, use_vertex_color=False,
        weighting_config=WeightingConfig(entropy_alpha=30.0, attention_layer=6, attention_step=0, min_weight=0.001),
        ss_weighting=True, ss_entropy_layer=9, ss_entropy_alpha=30.0, ss_warmup_steps=1)
    mesh = result["glb"]
    mesh = mesh.to_geometry() if isinstance(mesh, trimesh.Scene) else mesh
    mesh.apply_scale(float(result["scale"].reshape(-1)[0]))  # SAM 3D's metric scale, from view 0's depth
    to_origin, _ = trimesh.bounds.oriented_bounds(mesh)
    mesh.apply_transform(to_origin)
    return mesh


def render(mesh: trimesh.Trimesh, tile: int):
    """Four views from above the object's corners, with x/y/z drawn along the bounding box edges."""
    device = "cuda"
    texture = torch.from_numpy(np.asarray(mesh.visual.material.baseColorTexture.convert("RGB"))).float() / 255
    faces = torch.from_numpy(mesh.faces).long()
    meshes = Meshes([torch.from_numpy(mesh.vertices).float()], [faces], TexturesUV(
        [texture], [faces], [torch.from_numpy(mesh.visual.uv).float()])).to(device)
    lo, hi = mesh.bounds
    distance = 0.65 * np.linalg.norm(hi - lo) / np.tan(np.radians(15))  # fov 30
    corner = torch.tensor(np.array([lo] + [np.where(np.arange(3) == i, hi, lo) for i in range(3)]),
                          dtype=torch.float32, device=device)  # min corner, then its far end along x, y, z
    tiles = []
    for azimuth in (45, 135, 225, 315):
        a, e = np.radians(azimuth), np.radians(30)
        eye = distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        R, T = look_at_view_transform(eye=[eye.tolist()], up=((0, 0, 1),), device=device)
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=30, znear=0.01, zfar=10, device=device)
        renderer = MeshRenderer(
            MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=tile, bin_size=0)),
            HardPhongShader(device=device, cameras=cameras, blend_params=BlendParams(background_color=(1, 1, 1)),
                            lights=PointLights(location=[eye.tolist()], ambient_color=((0.6,) * 3,),
                                               diffuse_color=((0.4,) * 3,), specular_color=((0,) * 3,),
                                               device=device)))
        with torch.no_grad():
            image = (renderer(meshes)[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8).copy()
            px = cameras.transform_points_screen(corner, image_size=((tile, tile),))[:, :2].cpu().numpy()
        origin = tuple(int(v) for v in px[0])
        for i, name in enumerate("xyz"):
            end = tuple(int(v) for v in px[i + 1])
            cv2.arrowedLine(image, origin, end, AXIS_COLORS[i], 2, cv2.LINE_AA, tipLength=0.08)
            cv2.putText(image, name, (end[0] + 4, end[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AXIS_COLORS[i], 2,
                        cv2.LINE_AA)
        tiles.append(image)
    return tiles


def input_crop(rgb, mask, tile):
    ys, xs = np.nonzero(mask)
    half = int(0.75 * max(np.ptp(ys), np.ptp(xs))) + 10
    image = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
    image = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    y, x = (ys.min() + ys.max()) // 2 + half, (xs.min() + xs.max()) // 2 + half
    return cv2.resize(image[y - half:y + half, x - half:x + half], (tile, tile), interpolation=cv2.INTER_AREA)


def preview(mesh, views, tile: int) -> np.ndarray:
    """Row 1: the input views and the dimensions; row 2: four renders. RGB."""
    text = np.full((tile, tile, 3), 255, np.uint8)
    lines = [("Dimensions", (0, 0, 0))] + [(f"{n} = {e * 1000:.1f} mm", c)
                                           for n, e, c in zip("xyz", mesh.extents, AXIS_COLORS)]
    for k, (line, color) in enumerate(lines):
        cv2.putText(text, line, (16, 50 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    crops = [input_crop(rgb, mask, tile) for rgb, mask, _ in views][:3]
    row1 = crops + [np.full((tile, tile, 3), 255, np.uint8)] * (3 - len(crops)) + [text]
    return np.vstack([np.hstack(row1), np.hstack(render(mesh, tile))])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int, default=25)
    parser.add_argument("--stage2-steps", type=int, default=25)
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--edge-px", type=int, default=0, help="Drop depth this close inside the mask edge")
    args = parser.parse_args()

    view_dirs = sorted(args.views.glob("view_*"), key=lambda p: int(p.name.split("_")[1]))
    views = [load_view(p, args.edge_px) for p in view_dirs]
    mesh = reconstruct(views, args)
    args.out.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out / "mesh.glb")
    cv2.imwrite(str(args.out / "preview.png"), cv2.cvtColor(preview(mesh, views, args.tile), cv2.COLOR_RGB2BGR))
    print(f"mesh {len(mesh.vertices)} vertices, extents {np.round(mesh.extents * 1000, 1)} mm, "
          f"peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
