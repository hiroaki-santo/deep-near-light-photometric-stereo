#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Hiroaki Santo

import torch
import torch.nn as nn


def conv(batchNorm, c_in, c_out, k=3, stride=1, pad=-1):
    pad = (k - 1) // 2 if pad < 0 else pad
    # print('Conv pad = %d' % (pad))
    if batchNorm:
        print('=> convolutional layer with bachnorm')
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(c_out),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=pad, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


def deconv(c_in, c_out):
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1, inplace=True)
    )


class FeatExtractor(nn.Module):
    def __init__(self, c_in, batchNorm=False):
        super(FeatExtractor, self).__init__()
        self.conv1 = conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.conv4 = conv(batchNorm, 128, 256, k=3, stride=2, pad=1)
        self.conv5 = conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = deconv(256, 128)
        self.conv7 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.reshape(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False):
        super(Regressor, self).__init__()
        self.deconv1 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv2 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = deconv(128, 64)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class ReflectionEstimator(nn.Module):
    def __init__(self, batchNorm=False, c_out=3):
        super(ReflectionEstimator, self).__init__()

        self.deconv1 = conv(batchNorm, 256, 128, k=3, stride=1, pad=1)
        self.deconv2 = conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv3 = deconv(128, 64)
        self.est_reflection = self._make_output(64, c_out, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x):
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        refmaps = self.est_reflection(out)
        refmaps = nn.functional.relu(refmaps)  # positive-value
        return refmaps


class PsFcnReflectance(nn.Module):
    def __init__(self, batchNorm, c_in, device):
        super().__init__()

        self.device = device

        self.extractor = FeatExtractor(batchNorm=batchNorm, c_in=c_in + 3)
        self.regressor = Regressor(batchNorm=batchNorm)
        self.reflection_estimator = ReflectionEstimator(batchNorm=batchNorm, c_out=c_in)

    def forward(self, x):
        img, light = x
        # img.shape: [b, l, m, n, c_in]
        # light.shape: [b, l, m, n, 3]
        batch_size, light_num, m, n, ch_in = img.shape
        assert light.shape == (batch_size, light_num, m, n, 3)

        img, mess_normalization = self._normalize_input_img(img)
        feats = []
        for l in range(light_num):
            net_in = torch.cat([img[:, l], light[:, l]], -1)
            net_in = net_in.permute([0, 3, 1, 2])  # channel-first
            feat, shape = self.extractor(net_in)
            feats.append(feat)

        feat_fused, _ = torch.stack(feats, 1).max(1)
        normal = self.regressor(feat_fused, shape)

        reflectance_map = []
        for feat in feats:
            feat = feat.view(shape[0], shape[1], shape[2], shape[3])
            net_in = torch.cat([feat, feat_fused.view(shape[0], shape[1], shape[2], shape[3])], 1)
            reflectance = self.reflection_estimator(net_in)
            # [b, c, m, n]
            reflectance_map.append(reflectance)

        reflectance_map = torch.stack(reflectance_map, dim=1)

        # [b, l, c, m, n]
        normal = normal.permute([0, 2, 3, 1])  # channel-last, [b, m, n, 3]
        reflectance_map = reflectance_map.permute([0, 1, 3, 4, 2])  # channel-last, [b, l, m, n, c]

        reflectance_map = reflectance_map * mess_normalization  # recover the input scaling

        return [normal, reflectance_map, feat_fused]

    @staticmethod
    def _normalize_input_img(img):

        mess_normalization = img.norm(dim=1, keepdim=True).expand_as(img)
        mess_normalization = torch.div(mess_normalization, img.shape[1] ** 0.5)

        mess_normalization[mess_normalization == 0] = 1.
        img = img / (mess_normalization + 1e-12)

        return img, mess_normalization

    def predict(self, inputs):
        self.eval()
        feed = self.parse_inputs(inputs)
        outputs = self(feed)

        return outputs

    def parse_inputs(self, inputs):
        M = inputs["M"]
        L = inputs["L"]
        mask = inputs["mask"]
        # numpy or torch.Tensor

        batch_size, light_num, m, n, c_in = M.shape
        assert L.shape == (batch_size, light_num, m, n, 3), L.shape
        assert mask.shape == (batch_size, m, n), mask.shape

        if not torch.is_tensor(M):
            M = torch.from_numpy(M).float().to(self.device)
        if not torch.is_tensor(L):
            L = torch.from_numpy(L).float().to(self.device)
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).float().to(self.device)

        norm = L.norm(dim=-1, keepdim=True).expand_as(L)
        L = L / torch.add(norm, 1e-12)

        maskm = mask.reshape([batch_size, 1, m, n, 1]).expand_as(M)
        M[maskm == 0] = 0.
        maskm = mask.reshape([batch_size, 1, m, n, 1]).expand_as(L)
        L[maskm == 0] = 0.

        return [M, L]

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.load_state_dict(ckpt)
