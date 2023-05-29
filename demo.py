# Written by Ukcheol Shin (shinwc159[at]gmail.com)
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from dataloader.MS2_dataset import DataLoader_MS2
from utils.utils import visualize_disp_as_numpy, visualize_depth_as_numpy, Raw2Celsius

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_dir' , type=str, default='./MS2dataset')
    parser.add_argument('--modality', type=str, default='rgb', help='sensor modality: [rgb, nir ,thr]')
    parser.add_argument('--seq_name', type=str, default='_2021-08-06-10-59-33', help='sequence name')
    parser.add_argument('--data_format', type=str, default='MonoDepth', help='[MonoDepth, StereoMatch, MultiViewImg]')
    return parser.parse_args()

def main():
    args = parse_args()

    dataset_dir = args.dataset_dir
    seq_name    = args.seq_name
    data_format = args.data_format
    modality    = args.modality

    if data_format == 'MonoDepth':
      sampling_step = 50
      set_length = 1
      set_interval = 1
    elif data_format == 'StereoMatch':
      sampling_step = 50
      set_length = 1
      set_interval = 1
    elif data_format == 'MultiViewImg':
      sampling_step = 50
      set_length = 3
      set_interval = 5

    dataset       =   DataLoader_MS2(
                      dataset_dir,
                      data_split = seq_name,                                  
                      data_format = data_format,
                      modality=modality,
                      sampling_step=sampling_step,
                      set_length=set_length,
                      set_interval=set_interval
                  )

    demo_loader   = DataLoader(dataset, 
                              batch_size=1,
                              shuffle=False, 
                              num_workers=1, 
                              drop_last=False)

    print('{} samples found for evaluation'.format(len(demo_loader)))

    for i, batch in enumerate(tqdm(demo_loader)):
      if data_format == 'MonoDepth':
          if modality == 'thr':
            img = Raw2Celsius(batch["tgt_image"])
          else:
            img = batch["tgt_image"].type(torch.uint8)

          depth_gt = batch["tgt_depth_gt"]

          plt.subplot(2,1,1)
          plt.imshow(img[0])
          plt.subplot(2,1,2)
          plt.imshow(visualize_depth_as_numpy(depth_gt[0]))
          plt.pause(0.5)

      elif data_format == 'StereoMatch':
          if modality == 'thr':
            imgL = Raw2Celsius(batch["tgt_left"])
            imgR = Raw2Celsius(batch["tgt_right"])
          else:
            imgL = batch["tgt_left"].type(torch.uint8)
            imgR = batch["tgt_right"].type(torch.uint8)

          disp_gt = batch["tgt_disp_gt"]

          plt.subplot(3,1,1)
          plt.imshow(imgL[0])
          plt.subplot(3,1,2)
          plt.imshow(imgR[0])
          plt.subplot(3,1,3)
          plt.imshow(visualize_disp_as_numpy(disp_gt[0]))
          plt.pause(0.5)

      elif data_format == 'MultiViewImg':
          if modality == 'thr':
            tgt_img  = Raw2Celsius(batch["tgt_image"])
            ref_imgs = [Raw2Celsius(img) for img in batch["ref_images"]]
          else:
            tgt_img  = batch["tgt_image"].type(torch.uint8)
            ref_imgs = [img.type(torch.uint8) for img in batch["ref_images"]]

          plt.subplot(3,1,1) 
          plt.imshow(ref_imgs[0][0]) # T-N
          plt.subplot(3,1,2)
          plt.imshow(tgt_img[0]) # T
          plt.subplot(3,1,3)
          plt.imshow(ref_imgs[1][0]) # T+N
          plt.pause(0.5)

if __name__ == '__main__':
    main()
