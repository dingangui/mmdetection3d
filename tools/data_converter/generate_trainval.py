import mmcv

waymo_infos_train = mmcv.load("./data/waymo/kitti_format/waymo_infos_train.pkl")
waymo_infos_val = mmcv.load("./data/waymo/kitti_format/waymo_infos_val.pkl")
mmcv.dump(waymo_infos_train + waymo_infos_val, "./data/waymo/kitti_format/waymo_infos_trainval.pkl")

print("trainval finished")