* File Structure of "MS2 dataset"
|--sync_data
    |-- <Seq name>
        |-- rgb, nir, thr
            |-- img_left
            |-- img_right
            |-- depth_gt
            |-- depth_refl_gt
            |-- odom
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
|-- test_day_list.txt
|-- test_night_list.txt
|-- test_rainy_list.txt
