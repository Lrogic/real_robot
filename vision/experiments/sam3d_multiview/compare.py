"""Render and score every result of a session: MV-SAM3D (all views) and SAM 3D (each view alone).

Each mesh is placed in the manipulator base frame with its own predicted pose and the reference
view's T_base_camera, so every column is rendered from the same directions at the same scale:
from the side the views were captured from, then 90, 180 and 270 deg around the vertical, then
from above. The camera looks at each mesh's own centre, so placement errors do not shift it out
of frame. Colours are the meshes' vertex colours (no texture baking), lit from the camera.

The first row shows the input crop(s) with the mask outline. Column titles give the oriented
bounding box (mm, longest first) and, per captured view, the median distance from that view's
masked depth points to the mesh. Every view counts, including ones a single-view result never
saw; views other than the reference also carry the move-mode calibration error (about 6 to 10 mm).

Run in the sam3d env from ~/research/sam-3d-objects (reconstruct.py does this at the end):

    /opt/conda-envs/sam3d/bin/python <this file> SESSION [--size 320]

Writes SESSION/comparison.png and SESSION/metrics.json.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LIDRA_SKIP_INIT", "true")  # as notebook/inference.py: sam3d_objects.init is not shipped

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402
from pytorch3d.ops import knn_points  # noqa: E402
from pytorch3d.renderer import (BlendParams, FoVPerspectiveCameras, HardPhongShader, MeshRasterizer,  # noqa: E402
                                MeshRenderer, PointLights, RasterizationSettings, TexturesVertex,
                                look_at_view_transform)
from pytorch3d.structures import Meshes  # noqa: E402
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer  # noqa: E402

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # SAM 3D's to_glb: v_glb = v_local @ this
CV_FROM_P3D = np.diag([-1.0, -1.0, 1.0])  # PyTorch3D camera axes <-> OpenCV camera axes (its own inverse)
FOV_DEG = 30.0
ANGLES = (("capture side", 0, 25), ("+90 deg", 90, 25), ("back (180 deg)", 180, 25), ("+270 deg", 270, 25),
          ("top", 0, 89))
DEVICE = "cuda"


def load_view(view: Path):
    rgb = cv2.cvtColor(cv2.imread(str(view / "rgb.png")), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(view / "obj1/mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    depth = np.load(view / "depth.npy").astype(np.float64)
    K = np.array(json.loads((view / "camera_intrinsics.json").read_text())["K"])
    T = np.array(json.loads((view / "camera_pose.json").read_text())["T_base_camera"])
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    pts = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return dict(rgb=rgb, mask=mask, T=T, observed_base=pts @ T[:3, :3].T + T[:3, 3])


def load_result(path: Path, views):
    """Mesh in the base frame (metres) with vertex colours in [0, 1]."""
    meta = json.loads((path / "meta.json").read_text())
    params = np.load(path / "params.npz")
    glb = trimesh.load(path / "result.glb")
    mesh = glb.to_geometry() if isinstance(glb, trimesh.Scene) else glb
    t = lambda k: torch.from_numpy(params[k]).float().reshape(1, -1)  # noqa: E731
    local = torch.from_numpy(mesh.vertices @ Z_UP_TO_Y_UP.T).float()
    p3d = SceneVisualizer.object_pointcloud(points_local=local[None], quat_l2c=t("rotation"),
                                            trans_l2c=t("translation"), scale_l2c=t("scale")).points_list()[0]
    T = views[meta["reference_view"]]["T"]
    base = (p3d.double().numpy() @ CV_FROM_P3D) @ T[:3, :3].T + T[:3, 3]
    visual = mesh.visual if mesh.visual.kind == "vertex" else mesh.visual.to_color()
    colors = np.asarray(visual.vertex_colors, dtype=np.float32)[:, :3] / 255.0
    return dict(name=path.name, meta=meta, verts=base, faces=np.asarray(mesh.faces), colors=colors,
                voxels=int(params["coords"].shape[0]))


def metrics(result, views):
    placed = trimesh.Trimesh(result["verts"], result["faces"], process=False)
    dims = np.sort(placed.bounding_box_oriented.primitive.extents)[::-1] * 1000
    verts = torch.from_numpy(result["verts"]).float().to(DEVICE)
    fit = {}
    for i, view in views.items():
        obs = torch.from_numpy(view["observed_base"]).float().to(DEVICE)
        d = knn_points(obs[None], verts[None], K=1).dists[0, :, 0].sqrt()
        fit[i] = round(d.median().item() * 1000, 1)
    return dict(method=result["meta"]["method"], input_views=result["meta"]["views"],
                reference_view=result["meta"]["reference_view"], seed=result["meta"]["seed"],
                voxels=result["voxels"], obb_mm=dims.round(1).tolist(), median_depth_to_mesh_mm=fit)


def capture_azimuth(views, centre):
    """Azimuth (deg, base frame) of the mean horizontal direction from the object to the cameras."""
    d = np.mean([v["T"][:3, 3] - centre for v in views.values()], 0)
    return float(np.degrees(np.arctan2(d[1], d[0])))


def render(result, az0, distance, size):
    centre = 0.5 * (result["verts"].min(0) + result["verts"].max(0))
    mesh = Meshes(verts=[torch.from_numpy(result["verts"] - centre).float()],
                  faces=[torch.from_numpy(result["faces"]).long()],
                  textures=TexturesVertex([torch.from_numpy(result["colors"])])).to(DEVICE)
    images = []
    for _, az, el in ANGLES:
        a, e = np.radians(az0 + az), np.radians(el)
        eye = distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        up = (0, 0, 1) if el < 80 else (-np.cos(a), -np.sin(a), 0)  # top view: capture side at the bottom
        R, T = look_at_view_transform(eye=[eye.tolist()], at=((0, 0, 0),), up=(up,))
        cameras = FoVPerspectiveCameras(R=R, T=T, fov=FOV_DEG, znear=0.01, zfar=10.0, device=DEVICE)
        renderer = MeshRenderer(
            MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=size, faces_per_pixel=1, bin_size=0)),
            HardPhongShader(device=DEVICE, cameras=cameras,
                            lights=PointLights(location=[eye.tolist()], ambient_color=((0.55,) * 3,),
                                               diffuse_color=((0.45,) * 3,), specular_color=((0,) * 3,), device=DEVICE),
                            blend_params=BlendParams(background_color=(1, 1, 1))))
        with torch.no_grad():
            images.append(renderer(mesh)[0, ..., :3].clamp(0, 1).cpu().numpy())
    return images


def input_crop(view, size):
    ys, xs = np.nonzero(view["mask"])
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = int(0.75 * max(ys.max() - ys.min(), xs.max() - xs.min())) + 10
    image = view["rgb"].copy()
    contours, _ = cv2.findContours(view["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (255, 0, 255), 1)
    image = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    y, x = int(cy) + half, int(cx) + half
    return cv2.resize(image[y - half:y + half, x - half:x + half], (size, size), interpolation=cv2.INTER_AREA)


def tiled(crops, size):
    n = int(np.ceil(np.sqrt(len(crops))))
    tile = size // n
    canvas = np.full((size, size, 3), 255, np.uint8)
    for k, crop in enumerate(crops):
        r, c = divmod(k, n)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = cv2.resize(crop, (tile, tile))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", type=Path)
    parser.add_argument("--size", type=int, default=320, help="Pixels per tile")
    args = parser.parse_args()

    views = {int(p.name.split("_")[1]): load_view(p) for p in sorted(args.session.glob("view_*"))
             if (p / "obj1/mask.png").is_file()}
    dirs = sorted((args.session / "results").iterdir(),
                  key=lambda p: (p.name != "mvsam3d", int(p.name.removeprefix("sam3d_view") or 0)
                                 if p.name.startswith("sam3d_view") else 0))
    results = [load_result(d, views) for d in dirs if (d / "meta.json").is_file()]
    table = {r["name"]: metrics(r, views) for r in results}
    (args.session / "metrics.json").write_text(json.dumps(table, indent=2) + "\n")
    for name, m in table.items():
        print(f"{name:14s} views {m['input_views']}  obb {m['obb_mm']} mm  "
              f"median depth-to-mesh per view {m['median_depth_to_mesh_mm']} mm")

    centre = np.mean([0.5 * (r["verts"].min(0) + r["verts"].max(0)) for r in results], 0)
    az0 = capture_azimuth(views, centre)
    diag = max(np.linalg.norm(r["verts"].max(0) - r["verts"].min(0)) for r in results)
    distance = 0.6 * diag / np.tan(np.radians(FOV_DEG / 2))  # same scale for every column

    rows = 1 + len(ANGLES)
    fig, axes = plt.subplots(rows, len(results), figsize=(3.2 * len(results), 3.2 * rows + 0.8), squeeze=False)
    for c, r in enumerate(results):
        used = r["meta"]["views"]
        crops = [input_crop(views[v], args.size) for v in used]
        column = [crops[0] if len(crops) == 1 else tiled(crops, args.size)] + render(r, az0, distance, args.size)
        m = table[r["name"]]
        title = ("MV-SAM3D, views " + ",".join(map(str, used)) if m["method"] == "mvsam3d"
                 else f"SAM 3D, view {used[0]}")
        fit = " ".join(f"v{i}:{d:g}" for i, d in m["median_depth_to_mesh_mm"].items())
        axes[0, c].set_title(f"{title}\n{' x '.join(f'{x:g}' for x in m['obb_mm'])} mm\ndepth fit {fit}",
                             fontsize=8)
        for row, image in enumerate(column):
            axes[row, c].imshow(image)
            axes[row, c].set_xticks([])
            axes[row, c].set_yticks([])
            if c == 0:
                axes[row, c].set_ylabel("input" if row == 0 else ANGLES[row - 1][0], fontsize=9)
    fig.suptitle(f"{args.session.name}: seed {results[0]['meta']['seed']}; renders at one common scale; "
                 "depth fit = median distance (mm) from each view's masked depth to the mesh", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(args.session / "comparison.png", dpi=110)
    print(f"Wrote {args.session / 'comparison.png'} and metrics.json")


if __name__ == "__main__":
    main()
