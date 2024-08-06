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
import glob

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
    Rs: np.array = None
    Ts: np.array = None
    Es: np.array = None
    camera_extrinsics: np.array = None
    camera_intrinsics: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    num_objects: int = 1
    endeffector: np.ndarray = None

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

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
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
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
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

def readMultiRigidSceneInfo(path, images, eval, eval_ratio=0.2):

    # metadata = np.load(os.path.join(input_dir, "metadata.npz"))
    # print(f"Loaded metadata with keys: {metadata.files}")
    # camera_intrinsics = metadata["camera_intrinsics"]
    # camera_extrinsics = metadata["camera_extrinsics_calib"]
    # transforms_history = metadata["transforms_history"]
    # marker_pos = metadata["p3d_calib"]

    # Read the metadata file
    metadata = np.load(os.path.join(path, "metadata.npz"))
    cam_intrinsics = metadata["camera_intrinsics"]
    cam_extrinsics = metadata["camera_extrinsics_calib"]
    endeffector = metadata["p3d_calib"][:3]
    transforms = metadata["transforms_history"]
    num_objects = transforms.shape[0]
    num_frames = transforms.shape[1]
    obj_id = 1
    dummy_image = Image.open(glob.glob(os.path.join(path, images, "*.png"))[0])
    width, height = dummy_image.size
    FovX = np.arctan(width / (2 * cam_intrinsics[0, 0])) * 2
    FovY = np.arctan(height / (2 * cam_intrinsics[1, 1])) * 2
    image_paths = sorted(glob.glob(os.path.join(path, images, "*.png")))

    num_eval = int(num_frames * eval_ratio)
    num_train = num_frames - num_eval
    
    # # Add error to camera extrinsics
    # cam_extrinsics[0, 3] += 0.04

    # Create the camera infos
    cam_infos = []
    for idx in range(num_frames):
        image_path = image_paths[idx]

        Rs = np.zeros((num_objects, 3, 3))
        Ts = np.zeros((num_objects, 3))
        Es = np.zeros((num_objects, 4, 4))
        for obj_id in range(num_objects):
            transform = transforms[obj_id, idx]
            # extrinsics_stabilized = np.linalg.inv(transform) @ cam_extrinsics
            world_to_cam = np.linalg.inv(cam_extrinsics) @ transform # np.linalg.inv(extrinsics_stabilized) # world_to_cam = inv(ext_stab) = inv(inv(transform) @ ext) = inv(ext) @ transform

            Rs[obj_id] = world_to_cam[:3, :3].T
            Ts[obj_id] = world_to_cam[:3, 3]

            Es[obj_id] = transform


        cam_infos.append(
            CameraInfo(
                uid=idx,
                R=Rs[0],
                T=Ts[0],
                FovY=FovY,
                FovX=FovX,
                image=Image.open(image_path),
                image_path=image_path,
                image_name=os.path.basename(image_path).split(".")[0],
                width=width,
                height=height,
                Rs=Rs,
                Ts=Ts,
                Es=Es,
                camera_extrinsics=np.linalg.inv(cam_extrinsics),
                camera_intrinsics=cam_intrinsics
            )
        )

    if eval:
        train_cam_infos = cam_infos[:num_train]
        test_cam_infos = cam_infos[num_train:]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "point_cloud.ply")
    if os.path.exists(ply_path):
        os.remove(ply_path)
    
    # Since this data set has no colmap data, we start with random points
    num_pts = 10_000 * num_objects
    scene_size = 0.1
    # print(f"Generating random point cloud ({num_pts})...")
    
    # We create random points inside the bounds of the synthetic Blender scenes
    coordiantes_demo = False
    if coordiantes_demo:
        num_pts -= (num_pts % 3)
        one_third = num_pts // 3
        xyz = np.random.random((num_pts, 3)) * scene_size - scene_size / 2
        xyz = np.zeros((num_pts, 3))
        xyz[:one_third, 0] = np.linspace(-1.3, 1.3, one_third)
        xyz[one_third:2*one_third, 1] = np.linspace(-1.3, 1.3, one_third)
        xyz[2*one_third:, 2] = np.linspace(-1.3, 1.3, one_third)

        colors = np.zeros((num_pts, 3))
        colors[:one_third, 0] = np.linspace(0, 1, one_third)
        colors[one_third:2*one_third, 1] = np.linspace(0, 1, one_third)
        colors[2*one_third:, 2] = np.linspace(0, 1, one_third)
        
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))
    else:
        xyz = np.random.random((num_pts, 3)) * scene_size - (scene_size / 2)
        shs = np.random.random((num_pts, 3)) / 255.0
        colors = SH2RGB(shs)
        pcd = BasicPointCloud(points=xyz, colors=colors, normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, colors * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           num_objects=num_objects,
                           endeffector=endeffector)



    return scene_info


def readColmapSceneInfo(path, images, eval, llffhold=8):
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
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

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
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
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
    "Blender" : readNerfSyntheticInfo,
    "MultiRigid" : readMultiRigidSceneInfo
}