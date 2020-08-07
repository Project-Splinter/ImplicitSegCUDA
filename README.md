# ImplicitSeg

A Pytorch Segmentation module through implicit way (support 2d and 3d)

## Install

```
# via pip
pip install git+https://github.com/Project-Splinter/ImplicitSegCUDA --upgrade

# via git clone
git clone https://github.com/Project-Splinter/ImplicitSegCUDA
cd human_inst_seg
python setup.py develop
```

Note to run `demo.py` with `--vis` option, you also need to additional dependence: `vtkplotter` and `trimesh` and `skimage`.


## Prepare Test data
First `mkdir ./data`, then download `image.png`([here](https://drive.google.com/file/d/1OhpoGcMuk5LVyZjCYd4DXyHg4yYjbbVP/view?usp=sharing)) and `sdf.pth`([here](https://drive.google.com/file/d/1YWoctFOpe8Murzf6TvPMvaa4lcMpt3eM/view?usp=sharing)), and put them under `./data`.

## Usage

```
# 2d
python test/check_seg2d.py --mask ./data/image.png --mode lossless --vis
python test/check_seg2d.py --mask ./data/image.png --mode topk --vis
# 3d
python test/check_seg3d.py --voxel ./data/sdf.pth --mode lossless --vis
python test/check_seg3d.py --voxel ./data/sdf.pth --mode topk --vis
```

**Note**: `Seg3dTopk` and `Seg3dLossless` are both instances of `nn.Module`, so you need to be carefull when you want to integrate this to other trainable model.