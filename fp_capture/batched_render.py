"""Build every object's refinement crops in one render call, even with different meshes.

Extends the batched mode in batched_refine.py: instead of one make_crop_data_batch
call per object, all meshes live in one MeshBank and nvdiffrast's range mode
renders one crop per object in a single rasterize call. Textures are packed into
one atlas with a one-texel wrap-around border, so bilinear sampling matches
FoundationPose's per-mesh 'wrap' sampling. Reproduces make_crop_data_batch and
nvdiffrast_render from the FoundationPose checkout without modifying them.
Import only after fp_capture.tracking has put the checkout on sys.path.
"""

from typing import List, Sequence

import kornia
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

import learning.training.predict_pose_refine as fp_refine
import Utils as fp_utils

ATLAS_PAD = 1  # texels of wrap-around border around each texture tile


class MeshBank:
    """All objects' centred meshes concatenated into one set of GPU buffers.

    Row i of every batch is object i. Objects with identical textures share one
    atlas tile. Meshes must have either a texture with UVs in [0, 1] or vertex colours.
    """

    def __init__(self, estimators: Sequence[object]):
        pos, normals, faces, owner, uvs, colors, textured = [], [], [], [], [], [], []
        tiles: List[torch.Tensor] = []
        tile_of_object = []
        ranges = []
        vertex_offset = face_offset = 0
        for index, est in enumerate(estimators):
            tensors = est.mesh_tensors
            n_vertices, n_faces = len(tensors["pos"]), len(tensors["faces"])
            pos.append(tensors["pos"])
            normals.append(tensors["vnormals"])
            faces.append(tensors["faces"] + vertex_offset)
            owner.append(torch.full((n_vertices,), index, dtype=torch.long, device="cuda"))
            ranges.append((face_offset, n_faces))
            if "tex" in tensors:
                if not torch.equal(tensors["uv_idx"], tensors["faces"]):
                    raise NotImplementedError("Batched render needs per-vertex UVs (uv_idx == faces)")
                uv = tensors["uv"]
                if uv.min() < 0 or uv.max() > 1:
                    raise NotImplementedError("Batched render needs UVs inside [0, 1]")
                tile = next((i for i, t in enumerate(tiles) if t.shape == tensors["tex"].shape
                             and torch.equal(t, tensors["tex"])), None)
                if tile is None:
                    tiles.append(tensors["tex"])
                    tile = len(tiles) - 1
                tile_of_object.append(tile)
                uvs.append(uv)
                colors.append(torch.zeros((n_vertices, 3), device="cuda"))
                textured.append(torch.ones((n_vertices, 1), device="cuda"))
            else:
                tile_of_object.append(None)
                uvs.append(torch.zeros((n_vertices, 2), device="cuda"))
                colors.append(tensors["vertex_color"].float())
                textured.append(torch.zeros((n_vertices, 1), device="cuda"))
            vertex_offset += n_vertices
            face_offset += n_faces

        self.num_objects = len(estimators)
        self.pos = torch.cat(pos).float()
        self.normals = torch.cat(normals).float()
        self.faces = torch.cat(faces).int().contiguous()
        self.owner = torch.cat(owner)
        # nvdiffrast range mode wants the (start, count) triangle ranges as a CPU int32 tensor.
        self.ranges = torch.tensor(ranges, dtype=torch.int32, device="cpu")
        self.vertex_color = torch.cat(colors)
        self.has_texture = bool(tiles)
        self.has_vertex_color = any(t is None for t in tile_of_object)
        self.textured = torch.cat(textured)
        self.atlas = None
        self.uv = torch.cat(uvs)
        if self.has_texture:
            self.atlas, self.uv = self._build_atlas(tiles, tile_of_object, uvs)
        self.diameters = torch.as_tensor([e.diameter for e in estimators], device="cuda",
                                         dtype=torch.float)
        self.diameters_f64 = torch.as_tensor([float(e.diameter) for e in estimators],
                                             device="cuda", dtype=torch.float64)

    @staticmethod
    def _build_atlas(tiles, tile_of_object, uvs):
        p = ATLAS_PAD
        height = max(t.shape[1] for t in tiles) + 2 * p
        width = sum(t.shape[2] + 2 * p for t in tiles)
        atlas = torch.zeros((1, height, width, 3), device="cuda", dtype=torch.float)
        origins = []
        x = 0
        for tile in tiles:
            h, w = tile.shape[1:3]
            padded = F.pad(tile.permute(0, 3, 1, 2), (p, p, p, p), mode="circular")
            atlas[:, :h + 2 * p, x:x + w + 2 * p] = padded.permute(0, 2, 3, 1)
            origins.append((x + p, p, w, h))
            x += w + 2 * p
        remapped = []
        for tile, uv in zip(tile_of_object, uvs):
            if tile is None:
                remapped.append(uv)
                continue
            ox, oy, w, h = origins[tile]
            # Keeps texel centres aligned: u' * width - 0.5 == ox + u * w - 0.5.
            remapped.append(torch.stack([(ox + uv[:, 0] * w) / width,
                                         (oy + uv[:, 1] * h) / height], dim=-1))
        return atlas.contiguous(), torch.cat(remapped).contiguous()


def _crop_window_tfs(poses, K, diameters, crop_ratio, out_size):
    """compute_crop_window_tf_batch(method='box_3d') with one diameter per row.

    FoundationPose's version runs in float64 (its diameter and K are numpy float64)
    and only casts when writing the 3x3 result; matching that keeps the
    nearest-neighbour xyz crops identical.
    """
    n = len(poses)
    radius = (diameters.double() * crop_ratio / 2).reshape(n, 1, 1)
    unit = torch.tensor([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                        dtype=torch.float64, device="cuda")
    pts = poses[:, :3, 3].double().reshape(n, 1, 3) + unit.reshape(1, 5, 3) * radius
    K_t = torch.as_tensor(K, dtype=torch.float64, device="cuda")
    projected = (K_t @ pts.reshape(-1, 3).T).T
    uvs = (projected[:, :2] / projected[:, 2:3]).reshape(n, -1, 2)
    center = uvs[:, 0]
    radius_px = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(n, -1).max(dim=-1)[0]
    left = (center[:, 0] - radius_px).round()
    right = (center[:, 0] + radius_px).round()
    top = (center[:, 1] - radius_px).round()
    bottom = (center[:, 1] + radius_px).round()
    tf = torch.eye(3, device="cuda")[None].expand(n, -1, -1).contiguous()
    tf[:, 0, 2] = -left
    tf[:, 1, 2] = -top
    scale = torch.eye(3, device="cuda")[None].expand(n, -1, -1).contiguous()
    scale[:, 0, 0] = out_size[0] / (right - left)
    scale[:, 1, 1] = out_size[1] / (bottom - top)
    return scale @ tf


def _render(bank: MeshBank, glctx, poses, K, H, W, bbox2d, output_size):
    """nvdiffrast_render(use_light=True) for row i = object i at poses[i], in one call."""
    owner = bank.owner
    glcam_in_cvcam = torch.as_tensor(fp_utils.glcam_in_cvcam, device="cuda", dtype=torch.float)
    projection = torch.as_tensor(
        fp_utils.projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100),
        device="cuda", dtype=torch.float).reshape(1, 4, 4)
    mtx = projection @ (glcam_in_cvcam[None] @ poses)

    l, t = bbox2d[:, 0], H - bbox2d[:, 1]
    r, b = bbox2d[:, 2], H - bbox2d[:, 3]
    crop = torch.eye(4, device="cuda")[None].expand(len(poses), 4, 4).contiguous()
    crop[:, 0, 0] = W / (r - l)
    crop[:, 1, 1] = H / (t - b)
    crop[:, 3, 0] = (W - r - l) / (r - l)
    crop[:, 3, 1] = (H - t - b) / (t - b)

    pos_homo = fp_utils.to_homo_torch(bank.pos)
    pos_clip = (mtx[owner] @ pos_homo[..., None])[..., 0]
    pos_clip = (pos_clip[:, None, :] @ crop[owner])[:, 0, :].contiguous()
    rot, trans = poses[owner, :3, :3], poses[owner, :3, 3]
    pts_cam = ((rot @ bank.pos[..., None])[..., 0] + trans).contiguous()
    normals_cam = (rot @ bank.normals[..., None])[..., 0]

    rast, _ = dr.rasterize(glctx, pos_clip, bank.faces, resolution=np.asarray(output_size),
                           ranges=bank.ranges)
    xyz_map, _ = dr.interpolate(pts_cam, rast, bank.faces)
    depth = xyz_map[..., 2]

    if bank.has_texture:
        texc, _ = dr.interpolate(bank.uv, rast, bank.faces)
        color = dr.texture(bank.atlas, texc, filter_mode="linear", boundary_mode="clamp")
        if bank.has_vertex_color:
            vc, _ = dr.interpolate(bank.vertex_color, rast, bank.faces)
            weight, _ = dr.interpolate(bank.textured, rast, bank.faces)
            color = color * weight + vc * (1 - weight)
    else:
        color, _ = dr.interpolate(bank.vertex_color, rast, bank.faces)

    light_dir_neg = -torch.tensor([0.0, 0.0, 1.0], device="cuda")
    diffuse = (F.normalize(normals_cam, dim=-1) * light_dir_neg).sum(dim=-1).clip(0, 1)[..., None]
    diffuse_map, _ = dr.interpolate(diffuse.contiguous(), rast, bank.faces)
    color = color * 0.8 + diffuse_map * color * 0.5

    color = color.clip(0, 1) * torch.clamp(rast[..., -1:], 0, 1)
    return (torch.flip(color, dims=[1]), torch.flip(depth, dims=[1]),
            torch.flip(xyz_map, dims=[1]))


@torch.inference_mode()
def make_crops_batched(refiner, bank: MeshBank, glctx, poses, rgb, depth, xyz_map, K):
    """make_crop_data_batch for row i = object i at poses[i], with a per-row mesh."""
    cfg = refiner.cfg
    render_size = cfg.input_resize
    H, W = depth.shape[:2]
    n = len(poses)
    tf_to_crops = _crop_window_tfs(poses, K, bank.diameters_f64, cfg["crop_ratio"],
                                   (render_size[1], render_size[0]))
    corners = torch.as_tensor(
        np.array([0, 0, cfg["input_resize"][0] - 1, cfg["input_resize"][1] - 1]).reshape(2, 2),
        device="cuda", dtype=torch.float)
    # Utils.transform_pts would pair tf i with corner i when n == 2, so map explicitly.
    crop_to_ori = tf_to_crops.inverse()
    bbox2d_ori = ((crop_to_ori[:, None, :2, :2] @ corners[None, :, :, None])[..., 0]
                  + crop_to_ori[:, None, :2, 2]).reshape(-1, 4)

    rgb_r, _, xyz_r = _render(bank, glctx, poses, K, H, W, bbox2d_ori, cfg["input_resize"])
    rgb_rs = rgb_r.permute(0, 3, 1, 2) * 255
    xyz_map_rs = xyz_r.permute(0, 3, 1, 2)
    warp = kornia.geometry.transform.warp_perspective
    rgbAs = rgb_rs
    if rgb_rs.shape[-2:] != cfg["input_resize"]:
        rgbAs = warp(rgb_rs, tf_to_crops, dsize=render_size, mode="bilinear", align_corners=False)
    xyz_mapAs = xyz_map_rs
    if xyz_map_rs.shape[-2:] != cfg["input_resize"]:
        xyz_mapAs = warp(xyz_map_rs, tf_to_crops, dsize=render_size, mode="nearest",
                         align_corners=False)
    rgbBs = warp(rgb.permute(2, 0, 1)[None].expand(n, -1, -1, -1), tf_to_crops,
                 dsize=render_size, mode="bilinear", align_corners=False)
    xyz_mapBs = warp(xyz_map.permute(2, 0, 1)[None].expand(n, -1, -1, -1), tf_to_crops,
                     dsize=render_size, mode="nearest", align_corners=False)

    pose_data = fp_refine.BatchPoseData(
        rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=None, normalBs=None,
        poseA=poses, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs,
        tf_to_crops=tf_to_crops, Ks=torch.as_tensor(K, device="cuda", dtype=torch.float).reshape(1, 3, 3),
        mesh_diameters=bank.diameters)
    return refiner.dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)


@torch.inference_mode()
def refine_poses_batched_render(refiner, bank: MeshBank, glctx, poses, rgb, depth, xyz_map,
                                K, iterations: int) -> torch.Tensor:
    """Return (N, 4, 4) refined centred-mesh poses; row i is object i of the bank."""
    cfg = refiner.cfg
    if cfg["trans_rep"] != "tracknet" or cfg["use_normal"]:
        raise NotImplementedError("Batched render supports tracknet translation without normals")
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    diameters = bank.diameters.reshape(-1, 1)
    for _ in range(iterations):
        data = make_crops_batched(refiner, bank, glctx, poses, rgb, depth, xyz_map, K)
        A = torch.cat([data.rgbAs, data.xyz_mapAs], dim=1).float()
        B = torch.cat([data.rgbBs, data.xyz_mapBs], dim=1).float()
        with torch.autocast("cuda", enabled=refiner.amp):
            output = refiner.model(A, B)
        output = {k: v.float() for k, v in output.items()}
        if cfg["normalize_xyz"]:
            trans_delta = output["trans"] * (diameters / 2)
        else:
            trans_normalizer = cfg["trans_normalizer"]
            if not isinstance(trans_normalizer, float):
                trans_normalizer = torch.as_tensor(list(trans_normalizer), device="cuda",
                                                   dtype=torch.float).reshape(1, 3)
            trans_delta = torch.tanh(output["trans"]) * trans_normalizer
        if cfg["rot_rep"] == "axis_angle":
            rot_delta = torch.tanh(output["rot"]) * cfg["rot_normalizer"]
            rot_mat_delta = fp_refine.so3_exp_map(rot_delta).permute(0, 2, 1)
        elif cfg["rot_rep"] == "6d":
            rot_mat_delta = fp_refine.rotation_6d_to_matrix(output["rot"]).permute(0, 2, 1)
        else:
            raise RuntimeError(f"Unknown rot_rep {cfg['rot_rep']}")
        poses = fp_utils.egocentric_delta_pose_to_pose(poses, trans_delta=trans_delta,
                                                       rot_mat_delta=rot_mat_delta)
    return poses


def track_batched_render(estimators: List[object], bank: MeshBank, rgb: np.ndarray,
                         depth: np.ndarray, K: np.ndarray, iterations: int) -> List[np.ndarray]:
    """Like batched_refine.track_batched, but builds all crops in one render call."""
    first = estimators[0]
    if any(e.pose_last is None for e in estimators):
        raise RuntimeError("Every estimator must be registered before tracking")
    if any(e.refiner is not first.refiner or e.glctx is not first.glctx for e in estimators):
        raise ValueError("Batched tracking needs estimators that share a refiner and glctx")
    if len(estimators) != bank.num_objects:
        raise ValueError("MeshBank was built for a different set of estimators")

    depth_t = torch.as_tensor(depth, device="cuda", dtype=torch.float)
    depth_t = fp_utils.erode_depth(depth_t, radius=2, device="cuda")
    depth_t = fp_utils.bilateral_filter_depth(depth_t, radius=2, device="cuda")
    K_t = torch.as_tensor(K, dtype=torch.float, device="cuda")
    xyz_map = fp_utils.depth2xyzmap_batch(depth_t[None], K_t[None], zfar=np.inf)[0]
    # Upload uint8 and convert on the GPU; same values as converting on the CPU first.
    rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).cuda().float()

    poses = torch.cat([e.pose_last.reshape(1, 4, 4).float() for e in estimators])
    poses = refine_poses_batched_render(first.refiner, bank, first.glctx, poses, rgb_t, depth_t,
                                        xyz_map, K, iterations)
    results = []
    for index, estimator in enumerate(estimators):
        estimator.pose_last = poses[index:index + 1]
        results.append((poses[index] @ estimator.get_tf_to_centered_mesh()).cpu().numpy())
    return results
