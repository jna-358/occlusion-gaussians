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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh, RGB2SH
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.image_utils import discrete_colors
import numpy as np

projection_mat = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]
], dtype=np.float32)

# Function to convert a 3x3 rotation matrix to a quaternion (w, x, y, z)
def rotation_matrix_to_quaternion(R):
    # Ensure R is a proper rotation matrix of shape (3, 3)
    if R.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix.")

    # Allocate space for the quaternion
    q = torch.empty(4, device=R.device, dtype=R.dtype)

    # Compute the trace of the matrix
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = torch.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S

    return q

# Computes the 3x3 rotation matrix from a quaternion (w, x, y, z)
def quaternion_to_rotation_matrix(q):
    # Ensure q is a proper quaternion of shape (4,)
    if q.shape != (4,):
        raise ValueError("Input must be a 4-dimensional vector.")

    # Allocate space for the rotation matrix
    R = torch.empty((3, 3), device=q.device, dtype=q.dtype)

    # Compute the rotation matrix
    R[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    R[0, 1] = 2*q[1]*q[2] - 2*q[0]*q[3]
    R[0, 2] = 2*q[1]*q[3] + 2*q[0]*q[2]
    R[1, 0] = 2*q[1]*q[2] + 2*q[0]*q[3]
    R[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    R[1, 2] = 2*q[2]*q[3] - 2*q[0]*q[1]
    R[2, 0] = 2*q[1]*q[3] - 2*q[0]*q[2]
    R[2, 1] = 2*q[2]*q[3] + 2*q[0]*q[1]
    R[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2

    return R

# Compute the product of a set of quaternions with a single quaternion (w, x, y, z)
def quaternion_multiply(q1, q2, left_multiply=True):
    # if left_multiply:
    # Left multiplication q1 * q2
    w = q1[:, 0]*q2[0] - q1[:, 1]*q2[1] - q1[:, 2]*q2[2] - q1[:, 3]*q2[3]
    x = q1[:, 0]*q2[1] + q1[:, 1]*q2[0] + q1[:, 2]*q2[3] - q1[:, 3]*q2[2]
    y = q1[:, 0]*q2[2] - q1[:, 1]*q2[3] + q1[:, 2]*q2[0] + q1[:, 3]*q2[1]
    z = q1[:, 0]*q2[3] + q1[:, 1]*q2[2] - q1[:, 2]*q2[1] + q1[:, 3]*q2[0]
    # else:
    #     # Right multiplication q2 * q1
    #     w = q2[0]*q1[:, 0] - q2[1]*q1[:, 1] - q2[2]*q1[:, 2] - q2[3]*q1[:, 3]
    #     x = q2[0]*q1[:, 1] + q2[1]*q1[:, 0] + q2[2]*q1[:, 3] - q2[3]*q1[:, 2]
    #     y = q2[0]*q1[:, 2] - q2[1]*q1[:, 3] + q2[2]*q1[:, 0] + q2[3]*q1[:, 1]
    #     z = q2[0]*q1[:, 3] + q2[1]*q1[:, 2] - q2[2]*q1[:, 1] + q2[3]*q1[:, 0]

    return torch.stack([w, x, y, z], dim=1)

def render_occlusion(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier=1.0, 
           override_color=False, 
           gaussians_l=None, 
           render_indices=None,
           endeffector=None,
           ee_thresh=0.01,
           overwrite_tforms=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    K = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=viewpoint_camera.FoVx, fovY=viewpoint_camera.FoVy).transpose(0,1).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.eye(4, device="cuda"), # viewpoint_camera.world_view_transform,
        projmatrix=K, # viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    dists_l = {obj_id: None for obj_id in range(pc.get_num_rigid_bodies)}

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # Apply world view transform to means and rotations
    rigid_body_indices = pc.get_rigid_body_indices
    num_rigid_bodies = pc.get_num_rigid_bodies
    
    means3D_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
    rotations_all = torch.zeros_like(pc.get_rotation, dtype=pc.get_rotation.dtype, device="cuda")
    for rigid_body_index in range(num_rigid_bodies):
        # if overwrite_tforms is not None:
        #     transform = overwrite_tforms[rigid_body_index-1] # Needs to be refactored
        # else:
        #     transform = torch.from_numpy(viewpoint_camera.Es[rigid_body_index]).float().cuda()
        # camera_extrinsics = torch.from_numpy(viewpoint_camera.camera_extrinsics).float().cuda()
        # if rigid_body_index == 0:
        #     world_to_cam = camera_extrinsics @ transform
        # else:
        #     world_to_cam = camera_extrinsics @ transform

        # R = world_to_cam[:3, :3]
        # T = world_to_cam[:3, 3]

        if rigid_body_index > 0:
            if overwrite_tforms is not None:
                transform = overwrite_tforms[rigid_body_index-1] # Needs to be refactored
            else:
                transform = torch.from_numpy(viewpoint_camera.Es[rigid_body_index]).float().cuda()
            camera_extrinsics = torch.from_numpy(viewpoint_camera.camera_extrinsics).float().cuda()
            if rigid_body_index == 0:
                world_to_cam = camera_extrinsics @ transform
            else:
                world_to_cam = camera_extrinsics @ transform
            R = world_to_cam[:3, :3]
            T = world_to_cam[:3, 3]
        else:
            R = torch.eye(3, device="cuda")
            T = torch.tensor([0.0, 0.0, 10.0], device="cuda")

        world_view_rot = R
        world_view_quat = rotation_matrix_to_quaternion(world_view_rot)

        mask = (rigid_body_indices == rigid_body_index).unsqueeze(1)
        means3D_single = (pc.get_xyz @ R.T) + T
        rotations_single = quaternion_multiply(pc.get_rotation, world_view_quat)

        means3D_all += means3D_single * mask
        rotations_all += rotations_single * mask

        means3D = means3D_all
        rotations = rotations_all
        opacity = pc.get_opacity
        scales = pc.get_scaling
        shs = pc.get_features

        render_mask = torch.zeros((pc.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
        if render_indices is not None:
            for render_index in render_indices:
                render_mask = render_mask | (rigid_body_indices == render_index)

            means3D = means3D[render_mask]
            rotations = rotations[render_mask]
            opacity = opacity[render_mask]
            scales = scales[render_mask] 
            shs = shs[render_mask]

    # # Color all points close to the endeffector as white, the rest as black
    # ee_dist = torch.sqrt(torch.sum((means3D - endeffector[None, :])**2, 1))
    # is_close = ee_dist < ee_thresh
    # shs[is_close, :, :] = 2.0
    # shs[~is_close, :, :] = -2.0

    # # Add additional gray points to the endeffector
    # means3D_ee = torch.randn((100, 3)).float().cuda()
    # means3D_ee /= torch.sqrt(torch.sum(means3D_ee**2, 1, keepdim=True))
    # means3D_ee = means3D_ee * ee_thresh + endeffector[None, :]
    # rotations_ee = torch.zeros((means3D.shape[0], 4)).float().cuda()
    # rotations_ee[:, 0] = 1.0
    # opacity_ee = torch.ones((means3D_ee.shape[0], 1)).float().cuda()
    # scales_ee = torch.ones((means3D_ee.shape[0], 3)).float().cuda() * 1e-3
    # shs_ee = torch.zeros((means3D_ee.shape[0], shs.shape[1], shs.shape[2])).float().cuda()
    # shs_ee[:, :, :] = 2.0

    # means3D = torch.cat([means3D, means3D_ee], dim=0)
    # rotations = torch.cat([rotations, rotations_ee], dim=0)
    # opacity = torch.cat([opacity, opacity_ee], dim=0)
    # scales = torch.cat([scales, scales_ee], dim=0)
    # shs = torch.cat([shs, shs_ee], dim=0)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = None,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_bin = torch.mean(rendered_image, 0) > 0.5
    num_visible = torch.sum(rendered_bin).item()
    visibility = num_visible > 5

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "distances": dists_l,
            "visibility": visibility,
            "num_visible": num_visible
            }


def render(viewpoint_camera, 
           pc : GaussianModel, 
           pipe, 
           bg_color : torch.Tensor, 
           scaling_modifier=1.0, 
           override_color=False, 
           gaussians_l=None, 
           render_indices=None,
           overwrite_tforms=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    K = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=viewpoint_camera.FoVx, fovY=viewpoint_camera.FoVy).transpose(0,1).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=torch.eye(4, device="cuda"), # viewpoint_camera.world_view_transform,
        projmatrix=K, # viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    dists_l = {obj_id: None for obj_id in range(pc.get_num_rigid_bodies)}

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    projected_origins = np.empty((pc.get_num_rigid_bodies, 2), dtype=np.float32)

    shared_gs = True
    if shared_gs:
        # Apply world view transform to means and rotations
        rigid_body_indices = pc.get_rigid_body_indices
        num_rigid_bodies = pc.get_num_rigid_bodies
        
        means3D_all = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, device="cuda")
        rotations_all = torch.zeros_like(pc.get_rotation, dtype=pc.get_rotation.dtype, device="cuda")
        for rigid_body_index in range(num_rigid_bodies):
            if rigid_body_index > 0:
                if overwrite_tforms is not None:
                    transform = overwrite_tforms[rigid_body_index-1] # Needs to be refactored
                else:
                    transform = torch.from_numpy(viewpoint_camera.Es[rigid_body_index]).float().cuda()
                camera_extrinsics = torch.from_numpy(viewpoint_camera.camera_extrinsics).float().cuda()
                if rigid_body_index == 0:
                    world_to_cam = camera_extrinsics @ transform
                else:
                    world_to_cam = camera_extrinsics @ transform
                R = world_to_cam[:3, :3]
                T = world_to_cam[:3, 3]
            else:
                R = torch.eye(3, device="cuda")
                T = torch.tensor([0.0, 0.0, 10.0], device="cuda")


            world_view_rot = R
            world_view_quat = rotation_matrix_to_quaternion(world_view_rot)

            mask = (rigid_body_indices == rigid_body_index).unsqueeze(1)
            means3D_single = (pc.get_xyz @ R.T) + T
            rotations_single = quaternion_multiply(pc.get_rotation, world_view_quat)
            # dists_l[rigid_body_index] = means3D_single[:, 2]

            means3D_all += means3D_single * mask
            rotations_all += rotations_single * mask

            # Project the coordinate origin to the camera
            origin_3d = (viewpoint_camera.Es[rigid_body_index])[:, [3]]
            origin_2d = viewpoint_camera.camera_intrinsics @ projection_mat @ np.linalg.inv(viewpoint_camera.camera_extrinsics) @ origin_3d
            origin_2d = origin_2d[:,0]
            origin_2d = origin_2d[:2] / origin_2d[2]
            projected_origins[rigid_body_index] = origin_2d

        means3D = means3D_all
        rotations = rotations_all
        opacity = pc.get_opacity
        scales = pc.get_scaling
        shs = pc.get_features

        render_mask = torch.zeros((pc.get_xyz.shape[0],), dtype=torch.bool, device="cuda")
        if render_indices is not None:
            for render_index in render_indices:
                render_mask = render_mask | (rigid_body_indices == render_index)

            means3D = means3D[render_mask]
            rotations = rotations[render_mask]
            opacity = opacity[render_mask]
            scales = scales[render_mask] 
            shs = shs[render_mask]

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    else:
        # Apply world view transform to means and rotations
        R, T = viewpoint_camera.Rs[0], viewpoint_camera.Ts[0]
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1).cuda()
        means3D_A = gaussians_l[0].get_xyz
        rotations_A = gaussians_l[0].get_rotation
        means3D_A = (means3D_A @ world_view_transform[:3, :3]) + world_view_transform[3, :3]
        world_view_rot = world_view_transform[:3, :3]
        world_view_quat = rotation_matrix_to_quaternion(world_view_rot)
        rotations_A = quaternion_multiply(rotations_A, world_view_quat)

        R, T = viewpoint_camera.Rs[1], viewpoint_camera.Ts[1]
        world_view_transform = torch.tensor(getWorld2View2(R, T, np.zeros(3), 1.0)).transpose(0, 1).cuda()
        means3D_B = gaussians_l[1].get_xyz
        rotations_B = gaussians_l[1].get_rotation
        means3D_B = (means3D_B @ world_view_transform[:3, :3]) + world_view_transform[3, :3]
        world_view_rot = world_view_transform[:3, :3]
        world_view_quat = rotation_matrix_to_quaternion(world_view_rot)
        rotations_B = quaternion_multiply(rotations_B, world_view_quat)

        means3D = torch.cat([means3D_A, means3D_B], dim=0)
        rotations = torch.cat([rotations_A, rotations_B], dim=0)
        opacity = torch.cat([gaussians_l[0].get_opacity, gaussians_l[1].get_opacity], dim=0)
        scales = torch.cat([gaussians_l[0].get_scaling, gaussians_l[1].get_scaling], dim=0)
        shs = torch.cat([gaussians_l[0].get_features, gaussians_l[1].get_features], dim=0)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros((gaussians_l[0].get_xyz.shape[0]+gaussians_l[1].get_xyz.shape[0], 
                                          gaussians_l[0].get_xyz.shape[1]), dtype=gaussians_l[0].get_xyz.dtype, 
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

       
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    # if override_color is None:
    #     if pipe.convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    #     else:
    #         if shared_gs:
    #             shs = pc.get_features
    #         else:
    #             shs = torch.cat([gaussians_l[0].get_features, gaussians_l[1].get_features], dim=0)
    # else:
    #     colors_precomp = override_color

    # Project the coordinate origin to the camera
    origin_2d = viewpoint_camera.camera_intrinsics @ projection_mat @ viewpoint_camera.camera_extrinsics @ np.array([0, 0, 0, 1])[:, None]
    origin_2d = origin_2d[:, 0]
    origin_2d = origin_2d[:2] / origin_2d[2]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = screenspace_points,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "distances": dists_l,
            "origin_2d": origin_2d,
            "projected_origins": projected_origins
            }
