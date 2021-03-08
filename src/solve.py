#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Hiroaki Santo

import argparse
import glob
import math
import os

import cv2
import matplotlib
import numpy as np
import scipy.ndimage
import skimage.transform
import torch
import tqdm

from model_near_ps import NearLightPS

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable


def output_results(N, depth_map, mask, N_gt, depth_gt, output_path, step):
    m, n = mask.shape

    assert N.shape == (m, n, 3), N.shape
    assert depth_map.shape == (m, n), depth_map.shape
    assert mask.shape == (m, n), mask.shape

    assert N_gt.shape == (m, n, 3), N_gt.shape
    assert depth_gt.shape == (m, n), depth_gt.shape

    depth_map = depth_map - np.median(depth_map.detach().cpu().numpy()[mask != 0]) + np.median(
        depth_gt.detach().cpu().numpy()[mask != 0])

    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    [ax.axis("off") for ax in axs.flatten()]

    n_img = (N.detach().cpu().numpy() + 1.) / 2. * 255
    n_img[mask == 0] = 255
    depth_img = depth_map.detach().cpu().numpy()
    depth_img[mask == 0] = np.nan

    ax = axs[0]
    ax.imshow(n_img.astype(np.uint8))

    ax = axs[1]
    im = ax.imshow(depth_img, vmin=depth_img[mask != 0].min(), vmax=depth_img[mask != 0].max(), cmap="gray")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    fig.savefig(os.path.join(output_path, "fig_{step:05}.png".format(step=step)), bbox_inches='tight')
    plt.close(fig)

    np.savez(os.path.join(output_path, "result_{step:05}.npz".format(step=step)), Nest=N.detach().cpu().numpy(),
             depth_map=depth_map.detach().cpu().numpy(), mask=mask)

    #####
    MangEmap = NearLightPS.ang_error(N, N_gt).detach().cpu().numpy()
    MangEmap[mask == 0] = 0.
    MAngE = MangEmap[mask != 0].mean()

    MAbsEmap = torch.abs(depth_map - depth_gt).detach().cpu().numpy()
    MAbsE = MAbsEmap[mask != 0].mean()

    print("MAngE", np.rad2deg(MAngE), "MAbsE", MAbsE)


def _padding(padm, padn, M, N, mask, depth):
    M = np.pad(M, ((0, 0), (0, padm), (0, padn), (0, 0)), "edge")
    mask = np.pad(mask, ((0, padm), (0, padn)), "constant", constant_values=0)
    depth = np.pad(depth, ((0, padm), (0, padn)), "edge")
    N = np.pad(N, ((0, padm), (0, padn), (0, 0)), "edge")
    return M, N, mask, depth


def _crop_by_mask(M, mask, N, depth_map, camera_center, margin=20):
    light_num, m, n, ch = M.shape
    assert mask.shape == (m, n), mask.shape
    assert N.shape == (3, m * n), N.shape
    assert depth_map.shape == (m, n), depth_map.shape

    m_min = np.min(np.argwhere(np.sum(mask, axis=1) != 0))
    m_max = np.max(np.argwhere(np.sum(mask, axis=1) != 0))
    n_min = np.min(np.argwhere(np.sum(mask, axis=0) != 0))
    n_max = np.max(np.argwhere(np.sum(mask, axis=0) != 0))

    if m_min < 0 + margin:
        m_min += margin
    if m_max > m - margin:
        m_max -= margin
    if n_min < 0 + margin:
        n_min += margin
    if n_max > n - margin:
        n_max -= margin

    mask_ = mask[m_min - margin:m_max + margin, n_min - margin:n_max + margin]
    depth_map_ = depth_map[m_min - margin:m_max + margin, n_min - margin:n_max + margin]
    M_ = M[:, m_min - margin:m_max + margin, n_min - margin:n_max + margin]

    m_, n_ = mask_.shape
    N_ = N.T.reshape(m, n, 3)
    N_ = N_[m_min - margin:m_max + margin, n_min - margin: n_max + margin]
    N_ = N_.reshape(m_ * n_, 3).T

    camera_center_ = (camera_center[0] - (m_min - margin), camera_center[1] - (n_min - margin))

    return M_, mask_, N_, depth_map_, camera_center_


def load_dataset(dataset_path, obj_name):
    p = os.path.join(dataset_path, obj_name)
    assert os.path.exists(p), p
    print("II DATA: ", obj_name)

    S = np.loadtxt(os.path.join(p, "S.txt"))
    light_num, _ = S.shape

    M = []
    for l in tqdm.tqdm(range(light_num)):
        img = cv2.imread(os.path.join(p, "0", "{light_index}.png".format(light_index=l)), -1)
        M.append(img[:, :, ::-1])

    M = np.array(M, dtype=float)
    ambient = cv2.imread(os.path.join(p, "0", "{light_index}.png".format(light_index=light_num)), -1)
    ambient = ambient[..., ::-1]

    M = M - np.tile(ambient[np.newaxis], [light_num, 1, 1, 1])
    M[M < 0] = 0.
    M /= M.max()

    light_num, m, n, c_in = M.shape
    half_angle = np.deg2rad(60)
    mu_ = - np.log(2) / np.log(np.cos(half_angle))
    mu = np.ones(shape=(light_num, 1)) * mu_

    D = np.tile(np.array([0., 0., 1]).reshape(1, 3), [light_num, 1])
    phi = np.ones(shape=(light_num, c_in))

    mask_path = os.path.join(p, "gt", "mask.png")
    mask = cv2.imread(mask_path)[:, :, 0]
    mask[mask != 0] = 1.

    N = np.load(os.path.join(p, "gt", "normal_gt.npy"))
    mask[N[:, :, 2] < 0] = 0.

    N_ = N.reshape(m * n, 3).T

    depth_ = np.load(os.path.join(p, "gt", "depth_gt.npy"))
    depth_[mask == 0] = depth_[mask != 0].max()
    avg_distance = np.median(depth_[mask != 0])

    intrinsic = np.loadtxt(os.path.join(p, "params_camera.txt"))

    pixel_mm = np.sqrt(7.1 ** 2 + 5.4 ** 2) / np.sqrt(3072 ** 2 + 2048 ** 2)  # FLIR
    pixel_mm = [pixel_mm, pixel_mm]

    focal_length = (intrinsic[0, 0] + intrinsic[1, 1]) / 2.
    camera_center = (intrinsic[1, 2], intrinsic[0, 2])
    focal_length = focal_length * abs(pixel_mm[0])

    # # crop by mask
    M, mask, N_, depth_, camera_center = _crop_by_mask(M, mask, N_, depth_, camera_center)

    light_num, m, n, c_in = M.shape

    assert N_.shape == (3, m * n), N_.shape
    N_ = N_.T.reshape(m, n, 3)

    assert S.shape == (light_num, 3), S.shape
    assert mask.shape == (m, n), mask.shape
    assert len(camera_center) == 2, camera_center
    assert len(pixel_mm) == 2, pixel_mm

    padm = (4 - m % 4) % 4
    padn = (4 - n % 4) % 4
    if padm != 0 or padn != 0:
        M, N_, mask, depth_ = _padding(padm, padn, M, N_, mask, depth_)
        light_num, m, n, c_in = M.shape

    return M, S, D, mu, phi, mask, N_, depth_, avg_distance, camera_center, focal_length, pixel_mm, obj_name


def main(output_path, epoch, lr, lams, dataset_path, obj_name, model_path, iter_in_loop=100,
         light_select_num=32, pyramid_level=4, function_tolerance=1e-6, scaling=600, gpu=0):
    torch.autograd.set_detect_anomaly(True)
    if gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{gpu}".format(gpu=gpu)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    M, S, D, mu, phi, mask, N_, depth_, avg_distance, camera_center, focal_length, pixel_mm, obj_name = load_dataset(
        dataset_path, obj_name)

    light_num, m, n, c_in = M.shape
    if scaling > 0 and m * n > scaling ** 2:
        scaling_factor = math.ceil(np.sqrt(m * n / scaling ** 2))
        M, mask, N_, depth_, camera_center, pixel_mm = down_sampling(scaling_factor, M, mask, N_, depth_,
                                                                     camera_center, pixel_mm)
        light_num, m, n, c_in = M.shape

    ###################################################################
    M_orig, mask_orig, N_orig, depth_orig, camera_center_orig, pixel_mm_orig = M, mask, N_, depth_, camera_center, pixel_mm
    epoch_orig = epoch
    output_path_orig = output_path

    depth_init = None

    lr = lr * (float(avg_distance) / focal_length * abs(pixel_mm[0]))  # scaled by pixel_mm
    lr *= 2.  # discount lr /= 2. at the first epoch
    for pyramid_scaling in (2 ** np.arange(0, pyramid_level))[::-1]:
        print("II pyramid_scaling:", pyramid_scaling)
        M, mask, N_, depth_, camera_center, pixel_mm = down_sampling(pyramid_scaling, M_orig, mask_orig, N_orig,
                                                                     depth_orig, camera_center_orig, pixel_mm_orig)
        light_num, m, n, _ = M.shape

        epoch = epoch_orig
        output_path = os.path.join(output_path_orig, "s{}".format(pyramid_scaling))
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        lr = lr / 2.

        padm = (4 - m % 4) % 4
        padn = (4 - n % 4) % 4
        if padm != 0 or padn != 0:
            M, N_, mask, depth_ = _padding(padm, padn, M, N_, mask, depth_)
            light_num, m, n, c_in = M.shape

        if depth_init is not None:
            depth_init = skimage.transform.resize(depth_init, output_shape=(m, n), order=1, mode="edge")
            depth_init = scipy.ndimage.gaussian_filter(depth_init, sigma=1.3, mode="nearest")

        else:
            depth_init = np.ones(shape=(m, n)) * avg_distance

        model_cfg = {"device": device, "c_in": 3, "batchNorm": False, "ckpt_path": model_path}
        NPS = NearLightPS(M=M, S=S, S_dir=D, S_mu=mu, S_phi=phi, mask=mask.reshape(m, n), depth_init=depth_init,
                          model_cfg=model_cfg, device=device, focal_length=focal_length, camera_center=camera_center,
                          pixel_mm=pixel_mm, lams=lams)

        N_gt = torch.from_numpy(N_).type(torch.float).to(NPS.device)
        depth_gt = torch.from_numpy(depth_).type(torch.float).to(NPS.device)
        last_loss = torch.from_numpy(np.array([np.inf])).type(torch.float).to(NPS.device)

        NPS.init_optimizer(lr=lr)
        print("II: start")
        for epoch_count in range(1, 1 + epoch):
            NPS.minimize(iter_in_loop=iter_in_loop, light_select_num=light_select_num)
            with torch.no_grad():
                N, rho, Mest, losses = NPS()
                loss = NPS.get_loss(losses)
            print(epoch_count, "epoch:",
                  "loss: {} => {}, {}".format(last_loss.item(), loss.item(), (last_loss - loss).item()))

            if (last_loss - loss).item() < function_tolerance:
                output_results(N=N.permute([1, 2, 0]), depth_map=NPS.depth_map, depth_gt=depth_gt, N_gt=N_gt,
                               mask=mask.reshape(m, n), output_path=output_path, step=epoch_count)
                break

            last_loss = loss

        depth_init = NPS.depth_map.detach().cpu().numpy()
        depth_init = depth_init[:m - padm, :n - padn]
    ########
    output_results(N=N.permute([1, 2, 0]), depth_map=NPS.depth_map, depth_gt=depth_gt, N_gt=N_gt,
                   mask=mask.reshape(m, n), output_path=output_path_orig, step=9999)


def down_sampling(scaling, M, mask, N_, depth_, camera_center, pixel_mm):
    assert scaling > 0, scaling
    assert int(scaling) == scaling, scaling

    print("DD orig:", M.shape, mask.shape, depth_.shape, N_.shape)
    import skimage.measure

    M = skimage.measure.block_reduce(M, block_size=(1, scaling, scaling, 1), func=np.mean)
    mask = skimage.measure.block_reduce(mask, block_size=(scaling, scaling), func=np.min)
    depth_ = skimage.measure.block_reduce(depth_, block_size=(scaling, scaling), func=np.mean)
    N_ = skimage.measure.block_reduce(N_, block_size=(scaling, scaling, 1), func=np.mean)
    N_ /= np.tile(np.linalg.norm(N_, axis=2, keepdims=True), [1, 1, 3]) + 1e-12

    camera_center = (camera_center[0] / float(scaling), camera_center[1] / float(scaling))
    pixel_mm = (pixel_mm[0] * scaling, pixel_mm[1] * scaling)

    print("DD down to:", M.shape, mask.shape, depth_.shape, N_.shape)
    return M, mask, N_, depth_, camera_center, pixel_mm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--output_path", "-o", type=str, default="../output/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.5)

    parser.add_argument("--model_path", type=str, help="Pre-trained model for distant-light PS",
                        default="../data/PsFcnReflectance.pth")

    parser.add_argument("--dataset_path", type=str, default="../data/dataset")
    parser.add_argument("--obj_name", type=str, default="CAN")

    parser.add_argument("--lams", type=float, nargs="+", help="Hyper-parameter for loss", default=[0.05])

    ARGS = parser.parse_args()
    if not os.path.exists(ARGS.output_path):
        os.makedirs(ARGS.output_path)

    main(gpu=ARGS.gpu, output_path=ARGS.output_path, model_path=ARGS.model_path, epoch=ARGS.epoch,
         lr=ARGS.lr, lams=ARGS.lams, dataset_path=ARGS.dataset_path, obj_name=ARGS.obj_name)
