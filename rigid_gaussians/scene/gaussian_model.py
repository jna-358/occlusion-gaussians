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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import random

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, min_num_points=2500):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.rigid_body_indices = torch.empty(0, dtype=torch.short) # NEEDS TO BE SAVED
        self.num_rigid_bodies = -1 # NEEDS TO BE SAVED
        self.min_num_points = min_num_points # NEEDS TO BE SAVED

        self.second_body_start = None
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def clone(self):
        the_clone = GaussianModel(self.max_sh_degree)
        
        the_clone.active_sh_degree = self.active_sh_degree
        the_clone._xyz = self._xyz.clone()
        the_clone._features_dc = self._features_dc.clone()
        the_clone._features_rest = self._features_rest.clone()
        the_clone._scaling = self._scaling.clone()
        the_clone._rotation = self._rotation.clone()
        the_clone._opacity = self._opacity.clone()
        the_clone.max_radii2D = self.max_radii2D.clone()
        the_clone.xyz_gradient_accum = self.xyz_gradient_accum.clone()
        the_clone.denom = self.denom.clone()
        the_clone.rigid_body_indices = self.rigid_body_indices.clone()
        
        the_clone.num_rigid_bodies = self.num_rigid_bodies
        the_clone.second_body_start = self.second_body_start
        the_clone.optimizer = self.optimizer
        the_clone.percent_dense = self.percent_dense
        the_clone.spatial_lr_scale = self.spatial_lr_scale

        return the_clone

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.rigid_body_indices,
            self.num_rigid_bodies,
            self.min_num_points
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        self.rigid_body_indices,
        self.num_rigid_bodies,
        self.min_num_points) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_second_body_start(self):
        return self.second_body_start
    
    @property
    def get_rigid_body_indices(self):
        return self.rigid_body_indices
    
    @property
    def get_num_rigid_bodies(self):
        return self.num_rigid_bodies

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, num_rigid_bodies : int):
        demo_coords = ["none", "axes", "origin"][0]

        if demo_coords == "origin":
            self.num_rigid_bodies = num_rigid_bodies
            self.spatial_lr_scale = spatial_lr_scale
            self.rigid_body_indices = torch.arange(num_rigid_bodies).cuda()

            points = torch.zeros((num_rigid_bodies, 3)).float().cuda()
            scales = torch.log((torch.ones((num_rigid_bodies, 3)) * 1e-3).float().cuda())
            colors = torch.ones((num_rigid_bodies, 3)).float().cuda()

            fused_color = RGB2SH(colors).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            rots = torch.zeros((points.shape[0], 4)).cuda()
            rots[:, 0] = 1
            opacities = inverse_sigmoid(0.1 * torch.ones((points.shape[0], 1), dtype=torch.float).cuda())

            self._xyz = nn.Parameter(points.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
            self.second_body_start = points.shape[0] // 2


        elif demo_coords == "axes":
            self.num_rigid_bodies = num_rigid_bodies
            self.spatial_lr_scale = spatial_lr_scale
            points_per_body = 250
            num_pts = points_per_body * num_rigid_bodies

            self.rigid_body_indices = torch.arange(num_pts).cuda() % num_rigid_bodies
            # self.rigid_body_indices[:] = 7

            # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

            scene_size = 0.01

            fused_point_cloud = np.random.random((num_pts, 3)) * scene_size - (scene_size / 2)

            static_scale  = 10.0
            fused_color = np.zeros((num_pts, 3))  #np.random.random((num_pts, 3)) / 255.0

            # fused_point_cloud = torch.tensor(fused_point_cloud).float().cuda()
            # fused_point_cloud[self.rigid_body_indices == 0] *= static_scale

            # Draw box for each body
            box_points = num_pts // (15 * num_rigid_bodies)
            fused_point_cloud[:,:] = 0.0
            fused_color[:,:] = 0.0
            scales = np.ones((num_pts, 3)) * 1e-3

            grid_brightness = 1.0
            
            box_point_cloud = np.zeros((15 * box_points, 3))
            box_colors = np.zeros((15 * box_points, 3))
            box_scales = np.zeros((15 * box_points, 3))
            for i_side, (x, y) in enumerate([(-1,-1), (-1,1), (1,-1), (1,1)]):
                box_point_cloud[i_side*box_points:(i_side+1)*box_points, 0] = x
                box_point_cloud[i_side*box_points:(i_side+1)*box_points, 1] = y
                box_point_cloud[i_side*box_points:(i_side+1)*box_points, 2] = np.linspace(-1.0, 1.0, box_points)
                box_colors[i_side*box_points:(i_side+1)*box_points, :] = grid_brightness

            for i_side, (y, z) in enumerate([(-1,-1), (-1,1), (1,-1), (1,1)]):
                i_side_ = i_side + 4
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 1] = y
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 2] = z
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 0] = np.linspace(-1.0, 1.0, box_points)
                box_colors[i_side_*box_points:(i_side_+1)*box_points, :] = grid_brightness

            for i_side, (x, z) in enumerate([(-1,-1), (-1,1), (1,-1), (1,1)]):
                i_side_ = i_side + 8
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 2] = z
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 0] = x
                box_point_cloud[i_side_*box_points:(i_side_+1)*box_points, 1] = np.linspace(-1.0, 1.0, box_points)
                box_colors[i_side_*box_points:(i_side_+1)*box_points, :] = grid_brightness

            box_point_cloud[12*box_points:13*box_points, 0] = np.linspace(0, 1.0, box_points)
            box_point_cloud[13*box_points:14*box_points, 1] = np.linspace(0, 1.0, box_points)
            box_point_cloud[14*box_points:15*box_points, 2] = np.linspace(0, 1.0, box_points)

            box_colors[12*box_points:13*box_points, 0] = 255.0
            box_colors[13*box_points:14*box_points, 1] = 255.0
            box_colors[14*box_points:15*box_points, 2] = 255.0

            axis_scale = 1e-2
            box_scales[12*box_points:13*box_points, 0] = axis_scale
            box_scales[13*box_points:14*box_points, 1] = axis_scale
            box_scales[14*box_points:15*box_points, 2] = axis_scale

            points_per_body = box_points * 15
            for i_obj in range(num_rigid_bodies):
                fused_point_cloud[i_obj::num_rigid_bodies][:box_point_cloud.shape[0]] = box_point_cloud
                fused_color[i_obj::num_rigid_bodies][:box_colors.shape[0]] = box_colors
                scales[i_obj::num_rigid_bodies][:box_scales.shape[0]] = box_scales


            fused_point_cloud *= scene_size / 2

            fused_color = torch.tensor(fused_color).float().cuda()
            fused_color = RGB2SH(fused_color).float().cuda()
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            fused_point_cloud = torch.tensor(fused_point_cloud).float().cuda()

            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001) # torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            scales = torch.log(torch.from_numpy(scales).float().cuda())
            rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float).cuda())

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
            self.second_body_start = fused_point_cloud.shape[0] // 2
        
        elif demo_coords == "none":
            self.num_rigid_bodies = num_rigid_bodies
            self.spatial_lr_scale = spatial_lr_scale
            points_per_body = 10_000
            num_pts = points_per_body * num_rigid_bodies

            self.rigid_body_indices = torch.arange(num_pts).cuda() % num_rigid_bodies
            # self.rigid_body_indices[:] = 7

            # fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

            scene_size = 0.4

            fused_point_cloud = np.random.random((num_pts, 3)) * scene_size - (scene_size / 2)

            static_scale  = 25.0
            fused_color = np.random.random((num_pts, 3)) / 255.0

            fused_point_cloud = torch.tensor(fused_point_cloud).float().cuda()
            fused_color = torch.tensor(fused_color).float().cuda()
            fused_point_cloud[self.rigid_body_indices == 0] *= static_scale
            fused_point_cloud[self.rigid_body_indices == 0, 2] *= 0.001 # Flatten background

            # fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001) # torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4)).cuda()
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float).cuda())

            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
            self.second_body_start = fused_point_cloud.shape[0] // 2



    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.denom = torch.zeros((self.get_xyz.shape[0], 1)).cuda()

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]


        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float).cuda().requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float).cuda().transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float).cuda().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float).cuda().requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float).cuda().requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _ensure_min_points(self, valid_points_mask):
        # num_points_missing = self.min_num_points - torch.sum(valid_points_mask)
        # if num_points_missing > 0:
        #     possible_points = torch.nonzero(~valid_points_mask).squeeze()
        #     selected_points = possible_points[torch.randperm(num_points_missing)]
        #     valid_points_mask[selected_points] = True
        # return valid_points_mask

        for rigid_body_index in range(self.num_rigid_bodies):
            rigid_body_positions_pre = torch.nonzero(self.rigid_body_indices == rigid_body_index).squeeze(0)
            rigid_body_positions_post = torch.nonzero(valid_points_mask & (self.rigid_body_indices == rigid_body_index)).squeeze(0)
            num_points_missing = self.min_num_points - rigid_body_positions_post.shape[0]
            num_points_pre = rigid_body_positions_pre.shape[0]
            if num_points_missing > 0 and num_points_pre > 0:
                possible_points = torch.nonzero(~valid_points_mask & (self.rigid_body_indices == rigid_body_index)).squeeze()
                selected_points = possible_points[torch.randperm(num_points_missing)]
                valid_points_mask[selected_points] = True
        
        return valid_points_mask

    def prune_points(self, mask):
        valid_points_mask = ~mask
        valid_points_mask = self._ensure_min_points(valid_points_mask)

        # reshaped_mask = valid_points_mask.view(-1, self.num_rigid_bodies)
        # valid_mask_equal = torch.repeat_interleave(torch.any(reshaped_mask, dim=1), self.num_rigid_bodies)
        # valid_points_mask = valid_mask_equal

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.rigid_body_indices = self.rigid_body_indices[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_rigid_body_indices):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.denom = torch.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
        self.rigid_body_indices = torch.cat((self.rigid_body_indices, new_rigid_body_indices), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points)).cuda()
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_rigid_body_indices = self.rigid_body_indices[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_rigid_body_indices)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_rigid_body_indices = self.rigid_body_indices[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_rigid_body_indices)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        # self.order_rigid_bodies()
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def order_rigid_bodies(self):
        subset_ids = [torch.nonzero(self.rigid_body_indices == i).squeeze() for i in range(self.num_rigid_bodies)]
        subset_ids = torch.cat(subset_ids)

        self._xyz = self._xyz[subset_ids]
        self._features_dc = self._features_dc[subset_ids]
        self._features_rest = self._features_rest[subset_ids]
        self._scaling = self._scaling[subset_ids]
        self._rotation = self._rotation[subset_ids]
        self._opacity = self._opacity[subset_ids]
        self.max_radii2D = self.max_radii2D[subset_ids]
        self.xyz_gradient_accum = self.xyz_gradient_accum[subset_ids]
        self.denom = self.denom[subset_ids]
        self.rigid_body_indices = self.rigid_body_indices[subset_ids]
        

