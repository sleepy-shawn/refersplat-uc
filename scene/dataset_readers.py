#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import random
import re
import hashlib

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    gt_mask: dict
    sentence: list
    category: list
    is_negative: list = []
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder,
                      use_negative_samples=False, perturb_variant="",
                      use_spatial_negatives=False,
                      spatial_held_out_phrases="",
                      spatial_held_out_action="none",
                      max_cross_scene_neg_per_frame=-1,
                      max_spatial_neg_per_frame=-1,
                      training_neg_variants="",
                      training_neg_target_ratio=-1.0):
    """Loading semantics:

    - Positives always come from the main JSON's `object[]`. The "main"
      JSON is `<scene>/json/...` normally, or `<scene>/json_perturb_<variant>/...`
      when `perturb_variant` is set (zero-shot eval mode).

    - Cross-scene negatives: if `use_negative_samples` and NOT
      perturb_variant, read from main JSON's `negative[]`.

    - Perturbation negatives at zero-shot eval time: if `perturb_variant`
      is set, read from main (perturb) JSON's `perturbed[]`. The
      `use_negative_samples` flag is IGNORED in that mode.

    - Plan B training-time spatial negatives: if `use_spatial_negatives` is
      True AND `perturb_variant` is empty, ALSO read
      `<scene>/json_perturb_spatial/{train,test}_json/<frame>.json[perturbed]`
      and append entries as `is_negative=True`. Independent from
      `use_negative_samples` (both can be on).

    - Held-out preposition filter (only applies to spatial perturbed
      entries): `spatial_held_out_phrases` is a comma-separated list of
      phrases (matched against `change.from`).
        * `spatial_held_out_action="exclude"`: drop entries whose
          `change.from` matches (training-time hold-out).
        * `spatial_held_out_action="include_only"`: keep ONLY entries
          whose `change.from` matches (eval-time held-out test).
        * `spatial_held_out_action="none"` (default): no filter.

    - Per-frame caps `max_cross_scene_neg_per_frame` and
      `max_spatial_neg_per_frame`: -1 means no cap. Applied after
      held-out filtering, before shuffle.
    """
    held_phrases = {p.strip() for p in spatial_held_out_phrases.split(",") if p.strip()}
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        match = re.search(r'\d+', extr.name)
        number = int(match.group(0))
        
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        parent_folder = Path(images_folder).parent
        mask_name=extr.name.replace('.jpg', '')
        gt_mask_folder = os.path.join(parent_folder, 'gt_mask')
        if not os.path.isdir(gt_mask_folder):
            gt_mask_folder = os.path.join(parent_folder, 'mask')
        if len(mask_name)==2:
           mask_name=int(mask_name)
           mask_name=f"frame_{mask_name:05d}"
        if perturb_variant:
            # zero-shot eval: read from json_perturb_<variant>/{train,test}_json/
            base = os.path.join(parent_folder, f'json_perturb_{perturb_variant}')
            cand_train = os.path.join(base, 'train_json', mask_name + '.json')
            cand_test = os.path.join(base, 'test_json', mask_name + '.json')
            json_folder = cand_train if os.path.exists(cand_train) else cand_test
        elif os.path.exists(os.path.join(parent_folder, 'json')):
            json_folder = os.path.join(parent_folder, 'json/train_json', mask_name + '.json') if os.path.exists(os.path.join(parent_folder, 'json/train_json', mask_name + '.json')) else os.path.join(parent_folder, 'json/test_json', mask_name + '.json')
        else:
            json_folder = os.path.join(parent_folder, 'train_json', mask_name + '.json') if os.path.exists(os.path.join(parent_folder, 'json/train_json', mask_name + '.json')) else os.path.join(parent_folder, 'test_json', mask_name + '.json')
        with open(json_folder, 'r') as f:
            json_data = json.load(f)
            #print(json_data)

        # entries: [seg_path_or_None, sentence, category, is_negative]
        mask_sentence=[]
        for object in json_data['object']:
            seg_path = os.path.join(gt_mask_folder, object['segmentation'])
            if not os.path.exists(seg_path):
                continue
            for i in range(len(object['sentence'])):
                mask_sentence.append([seg_path, object['sentence'][i], object['category'], False])

        def _passes_held_out(change_dict):
            # Held-out filter on spatial perturbed entries. Checks BOTH
            # `change.from` AND `change.to` because the perturbation
            # generator is single-direction (a phrase may appear only on
            # one side of the swap, e.g. "on the edge of" is always a
            # `to`-side replacement). A swap is "held out" if EITHER side
            # touches the held-out vocabulary.
            if spatial_held_out_action == "none" or not held_phrases:
                return True
            cf = change_dict.get('from', '') if isinstance(change_dict, dict) else ''
            ct = change_dict.get('to', '') if isinstance(change_dict, dict) else ''
            in_held = (cf in held_phrases) or (ct in held_phrases)
            if spatial_held_out_action == "exclude":
                return not in_held
            if spatial_held_out_action == "include_only":
                return in_held
            return True

        if perturb_variant:
            # Zero-shot eval mode. Source is the main perturb JSON's `perturbed`.
            spatial_negs = []
            for p in json_data.get('perturbed', []):
                sent = p.get('sentence')
                if not sent:
                    continue
                if perturb_variant == "spatial":
                    if not _passes_held_out(p.get('change', {})):
                        continue
                cat = p.get('category', 'negative')
                spatial_negs.append([None, sent, cat, True])
            random.shuffle(spatial_negs)
            if max_spatial_neg_per_frame >= 0:
                spatial_negs = spatial_negs[:max_spatial_neg_per_frame]
            mask_sentence.extend(spatial_negs)
        else:
            # Normal training mode. Optionally stack cross-scene + spatial negs.
            if use_negative_samples:
                cs_negs = []
                for neg in json_data.get('negative', []):
                    neg_cat = neg.get('category', 'negative')
                    for sent in neg.get('sentence', []):
                        cs_negs.append([None, sent, neg_cat, True])
                random.shuffle(cs_negs)
                if max_cross_scene_neg_per_frame >= 0:
                    cs_negs = cs_negs[:max_cross_scene_neg_per_frame]
                mask_sentence.extend(cs_negs)

            # Defensive guard: if "spatial" is already in training_neg_variants,
            # disable the legacy use_spatial_negatives path so spatial negatives
            # are not double-counted (codex review M1).
            _legacy_spatial_active = use_spatial_negatives and (
                "spatial" not in [v.strip() for v in (training_neg_variants or "").split(",") if v.strip()]
            )
            if _legacy_spatial_active:
                spatial_neg_json = os.path.join(parent_folder, 'json_perturb_spatial',
                                                'train_json', mask_name + '.json')
                if not os.path.exists(spatial_neg_json):
                    spatial_neg_json = os.path.join(parent_folder, 'json_perturb_spatial',
                                                    'test_json', mask_name + '.json')
                if os.path.exists(spatial_neg_json):
                    with open(spatial_neg_json, 'r') as sf:
                        sp_data = json.load(sf)
                    sp_negs = []
                    for p in sp_data.get('perturbed', []):
                        sent = p.get('sentence')
                        if not sent:
                            continue
                        if not _passes_held_out(p.get('change', {})):
                            continue
                        cat = p.get('category', 'negative')
                        sp_negs.append([None, sent, cat, True])
                    random.shuffle(sp_negs)
                    if max_spatial_neg_per_frame >= 0:
                        sp_negs = sp_negs[:max_spatial_neg_per_frame]
                    mask_sentence.extend(sp_negs)

            # NEW: read multiple perturbation variants as training negatives.
            # Example: training_neg_variants = "attribute,category,spatial,borrow"
            # Each variant's `perturbed[]` entries are appended with
            # is_negative=True. Combines with `use_spatial_negatives` if the
            # caller specifies both (entries from "spatial" will appear once
            # via the legacy block and once via the new block, so it's the
            # caller's responsibility to set only ONE source for spatial).
            if training_neg_variants:
                variant_list = [v.strip() for v in training_neg_variants.split(',') if v.strip()]
                # Collect per-variant lists first; subsample / merge afterwards
                # so the target-ratio knob can stratified-trim across variants.
                collected_per_variant = {}
                for variant in variant_list:
                    var_json = os.path.join(parent_folder, f'json_perturb_{variant}',
                                            'train_json', mask_name + '.json')
                    if not os.path.exists(var_json):
                        var_json = os.path.join(parent_folder, f'json_perturb_{variant}',
                                                'test_json', mask_name + '.json')
                    if not os.path.exists(var_json):
                        continue
                    with open(var_json, 'r') as vf:
                        var_data = json.load(vf)
                    v_negs = []
                    for p in var_data.get('perturbed', []):
                        sent = p.get('sentence')
                        if not sent:
                            continue
                        # Apply held-out filter only to spatial entries.
                        if variant == 'spatial' and not _passes_held_out(p.get('change', {})):
                            continue
                        cat = p.get('category', 'negative')
                        v_negs.append([None, sent, cat, True])
                    # Deterministic per-frame, per-variant shuffle. hashlib (not
                    # Python's hash()) is required because hash() is PYTHONHASHSEED-
                    # salted per process and NOT cross-process deterministic.
                    _seed_var = int(hashlib.md5(f"{mask_name}|{variant}".encode()).hexdigest(), 16) & 0xFFFFFFFF
                    _rng_var = random.Random(_seed_var)
                    _rng_var.shuffle(v_negs)
                    # `max_spatial_neg_per_frame` is intentionally scoped to the
                    # 'spatial' variant only — applying it to all variants would
                    # silently cap attribute/category/borrow too.
                    if variant == 'spatial' and max_spatial_neg_per_frame >= 0:
                        v_negs = v_negs[:max_spatial_neg_per_frame]
                    collected_per_variant[variant] = v_negs

                # Determine final neg pool. With training_neg_target_ratio >= 0,
                # stratified-subsample across variants so per-frame
                #   neg / (pos + neg) ≈ training_neg_target_ratio.
                # Otherwise keep every variant's entries (legacy ~41% behaviour).
                #
                # `use_negative_samples` (legacy CS neg) bypasses this block by
                # design, so combining the two violates the target ratio. Fail
                # loudly instead of silently producing wrong data.
                if (training_neg_target_ratio is not None
                        and training_neg_target_ratio >= 0.0
                        and use_negative_samples):
                    raise ValueError(
                        "training_neg_target_ratio>=0 is incompatible with "
                        "use_negative_samples=True (legacy cross-scene "
                        "negatives bypass the ratio control). Disable one."
                    )
                pos_count = sum(1 for m in mask_sentence if not m[3])
                if training_neg_target_ratio is not None and training_neg_target_ratio >= 0.0:
                    if training_neg_target_ratio >= 1.0 or pos_count == 0:
                        target_neg = sum(len(v) for v in collected_per_variant.values())
                    else:
                        target_neg = int(round(training_neg_target_ratio * pos_count
                                               / (1.0 - training_neg_target_ratio)))
                    n_var = len(collected_per_variant)
                    if n_var > 0 and target_neg > 0:
                        # Ceil-divide quota per variant, then trim to total.
                        per_var_quota = (target_neg + n_var - 1) // n_var
                        for v in list(collected_per_variant.keys()):
                            collected_per_variant[v] = collected_per_variant[v][:per_var_quota]
                    combined = []
                    for v_negs in collected_per_variant.values():
                        combined.extend(v_negs)
                    # Deterministic combined shuffle.
                    _seed_combined = int(hashlib.md5(f"{mask_name}|combined".encode()).hexdigest(), 16) & 0xFFFFFFFF
                    _rng_combined = random.Random(_seed_combined)
                    _rng_combined.shuffle(combined)
                    combined = combined[:target_neg]
                    mask_sentence.extend(combined)
                else:
                    # Legacy: keep all entries from every variant.
                    for v_negs in collected_per_variant.values():
                        mask_sentence.extend(v_negs)
        # Deterministic per-camera shuffle so dataset readers produce
        # reproducible outputs across runs (hashlib for cross-process stability).
        _seed_final = int(hashlib.md5(mask_name.encode()).hexdigest(), 16) & 0xFFFFFFFF
        _rng_final = random.Random(_seed_final)
        _rng_final.shuffle(mask_sentence)
        if len(mask_sentence) == 0:
            gt_mask, sentence, category, is_negative = {}, [], [], []
        else:
            positive_paths = [p[0] for p in mask_sentence if p[0] is not None]
            unique_paths = list(set(positive_paths))
            gt_mask={path.split('/')[-1].split('_')[0].replace('.png', ''): Image.open(path) for path in unique_paths}
            sentence=[p[1] for p in mask_sentence]
            category=[p[2] for p in mask_sentence]
            is_negative=[p[3] for p in mask_sentence]

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, image_path=image_path, image_name=image_name, width=width, height=height, gt_mask=gt_mask, sentence=sentence, category=category, is_negative=is_negative)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

data_dict={
    'ramen':[6,24,60,65,81,119,128],
    'figurines':[83,97,146,179],
    'teatime':[2,25,43,107,129,140],
    'waldo':[19,35,67,105,162],
    'waldo_kitchen':[19,35,67,105,162],  # alias matching the README's documented directory name
}
def readColmapSceneInfo(path, images, eval, llffhold=8, use_negative_samples=False, perturb_variant="",
                        use_spatial_negatives=False,
                        spatial_held_out_phrases="",
                        spatial_held_out_action="none",
                        max_cross_scene_neg_per_frame=-1,
                        max_spatial_neg_per_frame=-1,
                        training_neg_variants="",
                        training_neg_target_ratio=-1.0):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    data_key=path.split('/')[-1]
    # reading_dir_F = "language_feature" if language_feature == None else language_feature
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=os.path.join(path, reading_dir),
        use_negative_samples=use_negative_samples,
        perturb_variant=perturb_variant,
        use_spatial_negatives=use_spatial_negatives,
        spatial_held_out_phrases=spatial_held_out_phrases,
        spatial_held_out_action=spatial_held_out_action,
        max_cross_scene_neg_per_frame=max_cross_scene_neg_per_frame,
        max_spatial_neg_per_frame=max_spatial_neg_per_frame,
        training_neg_variants=training_neg_variants,
        training_neg_target_ratio=training_neg_target_ratio,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    test_indices=data_dict[data_key]
    #import pdb; pdb.set_trace()
    train_cam_infos =[c for idx, c in enumerate(cam_infos) if int(c.image_name.split('_')[1]) not in test_indices ]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if int(c.image_name.split('_')[1]) in test_indices]
    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            # cam_name_F = os.path.join(path, frame["file_path"] + "") # TODO: extension?
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)


            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,  
                              image_path=image_path, 
                              image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
