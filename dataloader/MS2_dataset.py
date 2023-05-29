# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
import torch.utils.data as data
import numpy as np
import os.path as osp

import random
from path import Path
from utils.utils import load_as_float_img, load_as_float_depth

class DataLoader_MS2(data.Dataset):
    """A data loader where the files are arranged in this way:
        * Structure of "KAIST MS2 dataset"
        |--sync_data
            |-- <Seq name>
                |-- rgb, nir, thr
                    |-- img_left
                    |-- img_right
                |-- lidar
                    |-- left
                    |-- right
                |-- gps_imu
                    |-- data
                |-- calib.npy
        |--proj_depth
            |-- <Seq name>
                |-- rgb, nir, thr
                    |-- depth
                    |-- intensity
                    |-- depth_multi
                    |-- intensity_multi
                    |-- depth_filtered 
        |--odom
            |-- <Seq name>
                |-- rgb, nir, thr, odom
        |-- train_list.txt
        |-- val_list.txt
        |-- test_list.txt
    """
    def __init__(self, root, seed=None, data_split='train', modality='thr', \
                 data_format='MonoDepth', sampling_step=3, set_length=3, set_interval=1, opt=None):
        super(DataLoader_MS2, self).__init__()

        np.random.seed(seed)
        random.seed(seed)

        # read (train/val/test) data list
        self.root = Path(root)
        if data_split == 'train':
            data_list_file = self.root/'train_list.txt' 
        elif data_split == 'val':
            data_list_file = self.root/'val_list.txt' 
        elif data_split == 'test':
            data_list_file = self.root/'test_list.txt' 
        elif data_split == 'test_day':
            data_list_file = self.root/'test_day_list.txt' 
        elif data_split == 'test_night':
            data_list_file = self.root/'test_night_list.txt' 
        elif data_split == 'test_rain':
            data_list_file = self.root/'test_rainy_list.txt' 
        else: # when data_split is a specific sequence name
            data_list_file = data_split 

        # check if data_list_file has the .txt extension and create a list of folders
        if 'txt' in data_list_file:                
            self.seq_list = [seq_name[:-1] for seq_name in open(data_list_file)]
        else:
            self.seq_list = [data_list_file]

        self.modality = modality
        self.extrinsics = self.set_extrinsics()

        # determine which data getter function to use 
        if data_format == 'MonoDepth': # Monocular depth estimation, dict: {'img', 'depth'}
            self.data_getter = self.get_data_MonoDepth
            self.crawl_folders_depth(sampling_step, set_length, set_interval)
        elif data_format == 'StereoMatch':
            self.data_getter = self.get_data_StereoMatching
            self.crawl_folders_depth(sampling_step, set_length, set_interval)
        elif data_format == 'MultiViewImg':
            self.data_getter = self.get_data_MultiViewImg
            self.crawl_folders_depth(sampling_step, set_length, set_interval)
        elif data_format == 'Odometry':
            self.data_getter = self.get_data_Odometry
            self.crawl_folders_pose(sampling_step, set_length, set_interval)
        else:
            raise NotImplementedError(f"not supported type {data_format} in KAIST MS2 dataset.")

    def __getitem__(self, index):
        if isinstance(self.modality, list):
            results = {}
            if "rgb" in self.modality: results["rgb"] = self.data_getter(index, 'rgb')
            if "nir" in self.modality: results["nir"] = self.data_getter(index, 'nir')
            if "thr" in self.modality: results["thr"] = self.data_getter(index, 'thr')
            results["extrinsics"] = self.get_extrinsic()
            return results
        else:
            return self.data_getter(index, self.modality)

    def __len__(self):
        if isinstance(self.modality, list):
            return len(self.samples['rgb'])
        else:
            return len(self.samples[self.modality])

    def set_extrinsics(self) :
        # extrinsics matries are all same across the sequences, thus use the same values
        calib_path = osp.join(self.root, "sync_data", self.seq_list[0], "calib.npy")
        calib = np.load(calib_path, allow_pickle=True).item()

        ext_NIR2THR = np.concatenate([calib['R_nir2thr'], calib['T_nir2thr']*0.001], axis=1) # mm -> m scale conversion.
        ext_NIR2RGB = np.concatenate([calib['R_nir2rgb'], calib['T_nir2rgb']*0.001], axis=1)

        ext_THR2NIR = np.linalg.inv(np.concatenate([ext_NIR2THR, [[0,0,0,1]]],axis=0))
        ext_THR2RGB = np.matmul(np.concatenate([ext_NIR2RGB, [[0,0,0, 1]]],axis=0), ext_THR2NIR)

        ext_RGB2NIR = np.linalg.inv(np.concatenate([ext_NIR2RGB, [[0,0,0,1]]],axis=0))
        ext_RGB2THR = np.linalg.inv(ext_THR2RGB)

        extrinsics = {}
        extrinsics["NIR2THR"] = torch.as_tensor(ext_NIR2THR)
        extrinsics["NIR2RGB"] = torch.as_tensor(ext_NIR2RGB)

        extrinsics["THR2NIR"] = torch.as_tensor(ext_THR2NIR[0:3,:])
        extrinsics["THR2RGB"] = torch.as_tensor(ext_THR2RGB[0:3,:])

        extrinsics["RGB2NIR"] = torch.as_tensor(ext_RGB2NIR[0:3,:])
        extrinsics["RGB2THR"] = torch.as_tensor(ext_RGB2THR[0:3,:])
        return extrinsics

    def crawl_folders_depth(self, sampling_step, set_length, set_interval):
        # Args:
        # - sampling_step: sampling data from the crawled sample sets every N step. 
        # - set_length: control length of each sample set 
        #               if length = 1, each sample set only has time T sample, (i.e., single-view)
        #               if length > 1, each sample set has {T-N, ..., T, ..., T+N} (i.e., multi-view)
        # - set_interval: control interval between samples within sample set 
        #               if set_interval = N, each sample set has {T-N*M, ..., T-N, T, T+N, .., T+N*M}
        # if you want to do monocular vision task, set set_length = 1, set_interval = 1
        # if you want to do multi-view vision task, set set_length > 1 (odd number), set_interval >= 1
        # e.g., sampling_step = 2, set_length = 3, set_interval = 1
        # final sample sets = {{T-1, T, T+1}, {(T+2)-1, (T+2), (T+2)+1}, ...}

        # define shifts to select reference frames
        demi_length = (set_length-1)//2 
        shifts = [set_interval*i for i in range(-demi_length, demi_length + 1)]
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        # crawl the request modality list
        sensor_list = []
        if isinstance(self.modality, list):
            for modal in self.modality: 
                sensor_list.append(modal)
        else:
            sensor_list.append(self.modality)

        # iterate over sensor modalities
        sample_sets = {}
        for sensor in sensor_list:               
            # create an empty list to store sample set for current sensor modality
            sample_set = []
            for seq in self.seq_list: # iterate over each sequence
                calib_path = osp.join(self.root, "sync_data", seq, "calib.npy")
                calib = np.load(calib_path, allow_pickle=True).item()

                # read intrinsics and extrinsics (L/R) for current sensor modality
                if sensor == "rgb":
                    intrinsics     = calib['K_rgbL'].astype(np.float32)
                    extrinsics_L2R = np.concatenate([calib['R_rgbR'],calib['T_rgbR']*0.001], axis=1).astype(np.float32)
                    baseline       = abs(calib['T_rgbR'][0].astype(np.float32))*0.001 # convert to the meter scale
                elif sensor == "nir":
                    intrinsics     = calib['K_nirL'].astype(np.float32)
                    extrinsics_L2R = np.concatenate([calib['R_nirR'],calib['T_nirR']*0.001], axis=1).astype(np.float32)
                    baseline       = abs(calib['T_nirR'][0].astype(np.float32))*0.001 
                elif sensor == "thr":
                    intrinsics     = calib['K_thrL'].astype(np.float32)
                    extrinsics_L2R = np.concatenate([calib['R_thrR'],calib['T_thrR']*0.001], axis=1).astype(np.float32)
                    baseline       = abs(calib['T_thrR'][0].astype(np.float32))*0.001 

                img_list_left      = sorted((self.root/"sync_data"/seq/sensor/"img_left").files('*.png')) 
                img_list_right     = sorted((self.root/"sync_data"/seq/sensor/"img_right").files('*.png')) 

                # crawl the data path
                init_offset = demi_length*set_interval
                for i in range(init_offset, len(img_list_left)-init_offset):
                    # file name is all same across the folders (left, right, depth, depth_filterd, intensity, odom)
                    tgt_name  = img_list_left[i].name[:-4]
                    depth_in  = self.root/"proj_depth"/seq/sensor/"depth"/(tgt_name + '.png')
                    inten_in  = self.root/"proj_depth"/seq/sensor/"intensity"/(tgt_name + '.png')
                    depth_gt  = self.root/"proj_depth"/seq/sensor/"depth_filtered"/(tgt_name + '.png')

                    # create a dictionary containing information about the target and reference images, depth maps, etc.
                    sample = {'intrinsics': intrinsics, 
                              'tgt_img_left': img_list_left[i], 'tgt_img_right': img_list_right[i], 
                              'ref_imgs_left': [], 'ref_imgs_right': [],
                              'tgt_depth_in' : depth_in, 'tgt_depth_gt' : depth_gt,
                              'tgt_inten_in' : inten_in, 'ref_intens_in' : [],
                              'ref_depths_in' : [], 'ref_depths_gt' : [],
                              'baseline' : baseline, 'extrinsics_L2R' : extrinsics_L2R,
                             }

                    # Loop over the neighboring images to create the reference image list for the central image
                    for j in shifts:
                        ref_name       = img_list_left[i+j].name[:-4]
                        depth_in_  = self.root/"proj_depth"/seq/sensor/"depth"/(ref_name + '.png')
                        inten_in_  = self.root/"proj_depth"/seq/sensor/"intensity"/(ref_name + '.png')
                        depth_gt_  = self.root/"proj_depth"/seq/sensor/"depth_filtered"/(ref_name + '.png')

                        # append reference image, depth map, etc. to the sample dictionary
                        sample['ref_imgs_left'].append(img_list_left[i+j])
                        sample['ref_imgs_right'].append(img_list_right[i+j])
                        sample['ref_depths_in'].append(depth_in_)
                        sample['ref_intens_in'].append(inten_in_)
                        sample['ref_depths_gt'].append(depth_gt_)

                    sample_set.append(sample)

            # Subsampling the list of images according to the sampling step
            sample_set  = sample_set[0:-1:sampling_step]
            sample_sets[sensor] = sample_set
        self.samples = sample_sets

    def crawl_folders_pose(self, sampling_step, set_length, set_interval):
        # define shifts to select reference frames
        demi_length = (set_length-1)//2
        shift_range = np.array([set_interval*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

        # crawl the request modality list
        sensor_list = []
        if isinstance(self.modality, list):
            for modal in self.modality: 
                sensor_list.append(modal)
        else:
            sensor_list.append(self.modality)

        # iterate over each sensor modalitiy
        sample_sets = {}
        for sensor in sensor_list:               
            # create an empty list to store samples for this sensor modality
            sample_set = []
            for seq in self.seq_list: # iterate over each folder
                calib_path = osp.join(self.root, "sync_data", seq, "calib.npy")
                calib = np.load(calib_path, allow_pickle=True).item()

                img_list_left      = sorted((self.root/"sync_data"/seq/sensor/"img_left").files('*.png')) 
                img_list_right     = sorted((self.root/"sync_data"/seq/sensor/"img_right").files('*.png')) 

                # construct N-snippet sequences (note: N=set_length)
                init_offset = demi_length*set_interval
                tgt_indices = np.arange(init_offset, len(img_list_left) - init_offset).reshape(-1, 1)
                snippet_indices = shift_range + tgt_indices
    
                for indices in snippet_indices :
                    sample = {'imgs' : [], 'poses' : []}
                    for i in indices :
                        tgt_name = img_list_left[i].name[:-4]
                        pose  = self.root/"odom"/seq/sensor/(tgt_name + '.txt')
                        sample['imgs'].append(img_list_left[i])
                        sample['poses'].append(pose)
                    sample_set.append(sample)

            # Subsampling the list of images according to the sampling step
            sample_set  = sample_set[0:-1:sampling_step]
            sample_sets[sensor] = sample_set

        self.samples = sample_sets

    def depth2disp(self, depth, focal, baseline):
        min_depth = 1e-3
        mask = (depth < min_depth) 

        disp = baseline * focal / (depth +1e-10)
        disp[mask] = 0.
        return disp

    def get_extrinsic(self):
        return self.extrinsics

    # For monocular depth estimation
    def get_data_MonoDepth(self, index, modality):
        sample = self.samples[modality][index]

        tgt_img = load_as_float_img(sample['tgt_img_left'])
        tgt_depth_gt = load_as_float_depth(sample['tgt_depth_gt'])/256.0

        result = {}
        result["tgt_image"]    = tgt_img
        result["tgt_depth_gt"] = tgt_depth_gt
        return result

    # For stereo matching 
    def get_data_StereoMatching(self, index, modality):
        sample = self.samples[modality][index]

        tgt_img_left  = load_as_float_img(sample['tgt_img_left'])
        tgt_img_right = load_as_float_img(sample['tgt_img_right'])
        tgt_depth_gt  = load_as_float_depth(sample['tgt_depth_gt'])/256.0

        result = {}
        result["tgt_left"]       = tgt_img_left
        result["tgt_right"]      = tgt_img_right
        result["tgt_disp_gt"]    = torch.as_tensor(self.depth2disp(tgt_depth_gt,\
                                     focal=torch.as_tensor(np.copy(sample['intrinsics'][0,0])),\
                                     baseline=torch.as_tensor(sample['baseline'])))
        return result

    # For self-supervised monocular depth estimation
    def get_data_MultiViewImg(self, index, modality):
        sample = self.samples[modality][index]

        tgt_img  = load_as_float_img(sample['tgt_img_left'])
        ref_imgs = [load_as_float_img(ref_img) for ref_img in sample['ref_imgs_left']]

        result = {}
        result["tgt_image"]     = tgt_img
        result["ref_images"]    = ref_imgs
        result["intrinsics"]    = np.copy(sample['intrinsics'])
        return result

    # For multi-view pose estimation
    def get_data_Odometry(self, index, modality):
        sample = self.samples[modality][index]

        tgt_imgs = [load_as_float_img(img) for img in sample['imgs']]
        pose    = np.stack([np.genfromtxt(pose).astype(np.float64).reshape(4,4)[:3,:] for pose in sample['poses']]) # 3x4

        # uncomment if you need pose w.r.t img 0 within the list "tgt_imgs"
        # first_pose = poses[0]
        # poses[:,:,-1] -= first_pose[:,-1]
        # compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

        result = {}
        result["image"]    = tgt_imgs
        result["pose"]     = torch.as_tensor(pose) # Abloste pose (w.r.t first image of the sequence)
        return result