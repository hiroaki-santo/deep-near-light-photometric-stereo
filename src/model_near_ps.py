#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019 Hiroaki Santo

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import tqdm


def build_ps_function(model_cfg):
    from model_distant_ps import PsFcnReflectance
    ckpt_path = model_cfg.pop("ckpt_path")
    model = PsFcnReflectance(**model_cfg)
    model.load_ckpt(ckpt_path)

    for p in model.parameters():
        p.requires_grad = False

    return model


# linear regression model
class NearLightPS(torch.nn.Module):
    def __init__(self, M, S, S_dir, S_mu, S_phi, mask, lams, model_cfg, camera_center, pixel_mm, focal_length,
                 depth_init, dtype=torch.float, device="cuda:0"):
        super(NearLightPS, self).__init__()

        light_num, m, n, ch = M.shape
        self.lams = lams

        self.pixel_mm = pixel_mm
        self.focal_length = focal_length
        self.camera_center = camera_center

        assert len(self.camera_center) == 2, self.camera_center

        assert S.shape == (light_num, 3), S.shape
        assert S_dir.shape == (light_num, 3), S_dir.shape
        assert S_mu.shape == (light_num, 1), S_mu.shape
        assert S_phi.shape == (light_num, ch), S_phi.shape
        assert mask.shape == (m, n), mask.shape
        mask[mask != 0] = 1.

        self.device = torch.device(device)

        self.register_buffer("M", torch.autograd.Variable(torch.from_numpy(M).type(dtype)))
        self.register_buffer("S", torch.autograd.Variable(torch.from_numpy(S).type(dtype)))
        self.register_buffer("S_dir", torch.autograd.Variable(torch.from_numpy(S_dir).type(dtype)))
        self.register_buffer("S_mu", torch.autograd.Variable(torch.from_numpy(S_mu).type(dtype)))
        self.register_buffer("S_phi", torch.autograd.Variable(torch.from_numpy(S_phi).type(dtype)))
        self.register_buffer("mask", torch.autograd.Variable(torch.from_numpy(mask).type(dtype)))

        self.depth_init = depth_init

        self.depth_map = torch.nn.Parameter(
            torch.autograd.Variable(torch.from_numpy(self.depth_init).type(dtype), requires_grad=True)
        )

        xx, yy = np.meshgrid(np.arange(n), np.arange(m))
        xx = -np.abs(pixel_mm[0]) * (self.camera_center[1] - xx)
        yy = np.abs(pixel_mm[1]) * (self.camera_center[0] - yy)
        xy_on_sensor = np.concatenate([xx[np.newaxis], yy[np.newaxis]])

        self.register_buffer("xy_on_sensor", torch.autograd.Variable(torch.from_numpy(xy_on_sensor).type(dtype)))

        self.model_cfg = model_cfg
        self.model = build_ps_function(model_cfg)

        self.to(self.device)

    def forward(self, light_indices=None):
        M = self.M
        mask = self.mask
        light_num, m, n, c_in = M.shape
        assert mask.shape == (m, n), mask.shape

        if light_indices is None:
            light_indices = torch.arange(light_num, device=self.device)
        else:
            light_indices = light_indices.flatten()

        M = M[light_indices]
        S = self.S[light_indices]
        S_dir = self.S_dir[light_indices]
        S_mu = self.S_mu[light_indices]
        S_phi = self.S_phi[light_indices]
        light_num, m, n, c_in = M.shape

        #

        light_directions, light_distances = self._light_direction_per_pix(S=S, normalize=True)

        radiant = self._light_radiant(S_dir=S_dir, S_mu=S_mu, light_directions=light_directions)
        radiant = radiant.reshape(light_num, m, n, 1).expand(light_num, m, n, c_in)
        radiant = S_phi.reshape(light_num, 1, 1, c_in).expand(light_num, m, n, c_in) * radiant

        falloff_factor = light_distances.reshape(light_num, m, n, 1).expand_as(M) ** 2

        Md = M * falloff_factor / radiant
        light_directions[:, 2] *= -1
        # camera => image coordinates system

        L = light_directions.permute([0, 2, 3, 1])  # channel-last

        Mfeed = Md.reshape(1, light_num, m, n, c_in)
        L = L.reshape(1, light_num, m, n, 3)
        maskfeed = mask.reshape(1, m, n)

        outputs = self.model.predict(inputs={"M": Mfeed, "L": L, "mask": maskfeed})

        N = outputs[0][0]
        reflectances = outputs[1][0]

        assert N.shape == (m, n, 3), N.shape
        N = N.permute([2, 0, 1])  # channel-first

        assert N.shape == (3, m, n), N.shape
        N = torch.nn.functional.normalize(N, p=2, dim=0)

        N_from_depth = self.depth2normal(self.depth_map)

        Nest = N + N_from_depth
        Nest = torch.nn.functional.normalize(Nest, p=2, dim=0)

        shading = torch.sum(Nest.reshape(1, 3, m, n).expand_as(light_directions) * light_directions, dim=1)

        Mest = radiant * reflectances / falloff_factor * shading.reshape(light_num, m, n, 1).expand_as(M)
        Mest = torch.max(torch.zeros_like(Mest), Mest)

        Mest = Mest / Mest.max()  # to avoid overflow
        Mest = Mest / (2 * torch.sqrt(torch.mean(Mest[:, self.mask != 0] ** 2)))
        M = M / (2 * torch.sqrt(torch.mean(M[:, self.mask != 0] ** 2)))
        global_scale = torch.max(torch.cat([Mest, M]))
        loss_m = (Mest / global_scale - M / global_scale) ** 2

        loss_m[:, self.mask == 0.] = 0.
        loss_m = torch.mean(loss_m[:, self.mask != 0])
        #
        loss_n = 1. - torch.sum(N * N_from_depth, dim=0)
        # loss_n = torch.sum((N - N_from_depth) ** 2, dim=0)

        loss_n[self.mask == 0] = 0.
        loss_n = torch.mean(loss_n[self.mask != 0])

        return Nest, reflectances, Mest, [loss_m, loss_n]

    def init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def minimize(self, iter_in_loop, light_select_num):

        light_num, m, n, _ = self.M.shape

        for i in tqdm.tqdm(range(iter_in_loop)):
            indices = torch.randperm(light_num, dtype=torch.long, device=self.device)[:min(light_num, light_select_num)]
            N, rho, rendered, losses = self(indices)
            loss = self.get_loss(losses)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def get_loss(self, losses):
        loss = (1. - np.sum(self.lams)) * losses[0] + self.lams[0] * losses[1]
        return loss

    def xy_map(self, depth_map):
        xy = torch.stack([
            torch.mul(torch.div(depth_map, self.focal_length), self.xy_on_sensor[0]),
            torch.mul(torch.div(depth_map, self.focal_length), self.xy_on_sensor[1])
        ])

        return xy

    def _light_direction_per_pix(self, S, normalize=True):
        light_num, _ = S.shape
        m, n = self.depth_map.shape

        xyz = torch.cat([self.xy_map(self.depth_map), self.depth_map.reshape(1, m, n)], dim=0)
        xyz = xyz.reshape([1, 3, m, n])
        xyz = xyz.expand(light_num, 3, m, n)
        S = S.reshape(light_num, 3, 1, 1)
        S = S.expand_as(xyz)

        light_directions = S - xyz  # [f, 3, m, n]
        light_distances = light_directions.norm(p=2, dim=1)  # [f, m, n]
        if normalize:
            norm = light_directions.norm(p=2, dim=1, keepdim=True).expand_as(light_directions)
            norm = torch.add(norm, 1e-12)
            light_directions = light_directions / norm

        return light_directions, light_distances

    @classmethod
    def _light_radiant(cls, S_dir, S_mu, light_directions):
        light_num, _, m, n = light_directions.shape  # surface to light in camera coordinates system
        assert S_dir.shape == (light_num, 3), S_dir.shape  # principle direction in camera coordinates system
        assert S_mu.shape == (light_num, 1), S_mu.shape

        dot = torch.sum(S_dir.reshape(light_num, 3, 1, 1).expand(light_num, 3, m, n) * -light_directions, dim=1)
        radiant = torch.pow(dot, S_mu.reshape(light_num, 1, 1).expand(light_num, m, n))
        return radiant

    def depth2normal(self, depth):
        m, n = depth.shape

        filter_x = torch.tensor([[0, 0., 0], [1., 0., -1.], [0., 0., 0.]]) / (2. * abs(self.pixel_mm[0]))
        filter_y = torch.tensor([[0, -1., 0.], [0., 0., 0], [0., 1, 0.]]) / (2. * abs(self.pixel_mm[1]))
        fx = filter_x.expand(1, 1, 3, 3).to(depth.device)
        fy = filter_y.expand(1, 1, 3, 3).to(depth.device)
        p = torch.nn.functional.conv2d(depth.reshape(1, 1, m, n), fx, stride=1, padding=1)[0]
        q = torch.nn.functional.conv2d(depth.reshape(1, 1, m, n), fy, stride=1, padding=1)[0]
        #
        nz = torch.div(depth + self.focal_length + (self.xy_on_sensor[0] * p[0] + self.xy_on_sensor[1] * q[0]),
                       self.focal_length)
        nz = nz.reshape(1, m, n)
        normal = torch.cat([-p, -q, nz], dim=0)

        normal = torch.nn.functional.normalize(normal, p=2, dim=0)

        return normal

    @staticmethod
    def ang_error(a, b):
        assert a.shape == b.shape, (a.shape, b.shape)
        assert a.shape[-1] == 3, a.shape

        norm = a.norm(p=2, dim=-1)
        norm[norm == 0] = 1.
        norm_a = norm

        norm = b.norm(p=2, dim=-1)
        norm[norm == 0] = 1.
        norm_b = norm

        dot = torch.sum(a * b, dim=-1) / (norm_a * norm_b)
        dot = torch.clamp(dot, -1., 1.)
        ang_error = torch.acos(dot)

        return ang_error
