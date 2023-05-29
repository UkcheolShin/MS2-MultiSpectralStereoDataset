# Multi-Spectral Stereo ($MS^2$) Outdoor Driving Dataset

This is the official github page of the $MS^2$ dataset described in the following paper.

This page provides a dataloader and simple python code for $MS^2$ dataset.

If you want to download the dataset and see the details, please visit the [dataset page](https://sites.google.com/view/multi-spectral-stereo-dataset).

 >Deep Depth Estimation from Thermal Image
 >
 >[Ukcheol Shin](https://ukcheolshin.github.io/), Jinsun Park, In So Kweon
 >
 >IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023
 >
 >[[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shin_Deep_Depth_Estimation_From_Thermal_Image_CVPR_2023_paper.pdf)] [[Dataset page](https://sites.google.com/view/multi-spectral-stereo-dataset)]

## Updates
- 2023.03.30: Open Github page.
- 2023.05.30: Release $MS^2$ dataset, dataloader, and demo code.

## $MS^2$ Dataset Specification
MS2 dataset provides:
- (Synchronized) Stereo RGB images / Stereo NIR images / Stereo thermal images
- (Synchronized) Stereo LiDAR scans / GPS/IMU navigation data
- Projected depth map (in RGB, NIR, thermal image planes)
- Odometry data (in RGB, NIR, thermal cameras, and LiDAR coordinates)

## Usage
1. Download the datasets and place them in 'MS2dataset' folder in the following structure:

```shell
MS2dataset
├── sync_data
│   ├── <Sequence Name1>
│   ├── <Sequence Name2>
│   ├── ...
│   └── <Sequence NameN>
├── proj_depth
│   ├── <Sequence Name1>
│   ├── <Sequence Name2>
│   ├── ...
│   └── <Sequence NameN>
└── odom
    ├── <Sequence Name1>
    ├── <Sequence Name2>
    ├── ...
    └── <Sequence NameN>
```

2. We provide a simple python code (demo.py) along with a dataloader to take a look at the provided dataset.
To run the code, you need any version of Pytorch library.
```bash
python demo.py --seq_name <Sequence Name> --modality rgb --data_format MonoDepth
python demo.py --seq_name <Sequence Name> --modality nir --data_format StereoMatch
python demo.py --seq_name <Sequence Name> --modality thr --data_format MultiViewImg
```
