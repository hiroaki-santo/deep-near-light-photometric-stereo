# Deep near-light photometric stereo

This is the project page for our ECCV 2020 paper: "Deep near-light photometric stereo for spatially varying
reflectances" by Hiroaki Santo, Michael Waechter, and Yasuyuki
Matsushita. [[paper]](http://cvl.ist.osaka-u.ac.jp/wp-content/uploads/2020/08/Santo_eccv2020.pdf)

If you use our paper or code for research purposes, please cite our paper:

```
@inproceedings{santo2020deep,
title = {Deep near-light photometric stereo for spatially varying reflectances},
booktitle = {European Conference on Computer Vision (ECCV)},
author = {Hiroaki Santo, Michael Waechter, Yasuyuki Matsushita},
year = {2020},
}
```

## Environment

* CUDA 9.0 (CuDNN 7)
* PyTorch (1.1.0)

## Dataset

* CAN
* CUP
* TURTLE

Please download from [here](https://www.dropbox.com/t/bQmg9BOdA7OabKeH) and extract them to `PATH/TO/DATASET`.

## How to run

```bash
python solve.py --dataset_path PATH/TO/DATASET --obj_name {CAN/CUP/TURTLE} --output_path PATH/TO/OUTPUT
```

In `PATH/TO/OUTPUT`, the estimated normal and depth maps are stored as a npz file.
