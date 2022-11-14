# Copyright (c) OpenMMLab. All rights reserved.
r"""Adapted from `Waymo to KITTI converter
    <https://github.com/caizhongang/waymo_kitti_converter>`_.
"""

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')
# from waymo_open_dataset.protos import camera_segmentation_pb2 as cs_pb2

from glob import glob
from os.path import join
from mmcv.utils import print_log

import mmcv
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils import frame_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
import tensorflow as tf
import numpy as np
if not tf.executing_eagerly():
  tf.compat.v1.enable_eager_execution()

import random
import cv2

def decode_single_panoptic_label_from_proto(
    segmentation_proto) -> np.ndarray:
  """Decodes a panoptic label from a CameraSegmentationLabel proto.
  Args:
    segmentation_proto: a CameraSegmentationLabel proto to be decoded.
  Returns:
    A 2D numpy array containing the per-pixel panoptic segmentation label.
  """
  return tf.io.decode_png(
      segmentation_proto.panoptic_label, dtype=tf.uint16).numpy()
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (255, 255, 128), (147, 211, 203),
               (150, 100, 100), (168, 171, 172), (146, 112, 198),
               (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
               (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
               (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
               (106, 154, 176),
               (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
               (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
               (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
               (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
               (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
               (146, 139, 141),
               (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
               (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
               (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
               (102, 102, 156), (250, 141, 255)]

def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            \n
            '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)
        
        
class Waymo2KITTI(object):
    """Waymo to KITTI converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 prefix,
                 workers=64,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = True # 此处 和 caizhongang 不一样, cai 为 true

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)
        self.test_mode = test_mode

        self.tfrecord_pathnames = sorted(
            glob(join(self.load_dir, '*.tfrecord')))

        self.label_save_dir = f'{self.save_dir}/label_'
        self.label_all_save_dir = f'{self.save_dir}/label_all'
        self.image_save_dir = f'{self.save_dir}/image_'
        self.calib_save_dir = f'{self.save_dir}/calib'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'
        self.depth_save_dir = f'{self.save_dir}/depth_'
        self.seg_save_dir = f'{self.save_dir}/seg_'
        self.seg_json_save_dir = f'{self.save_dir}/seg_json_'
        self.create_folder()

    def get_file_size(self, filePath):
        filePath = str(filePath)
        fsize = os.path.getsize(filePath)
        fsize = fsize/float(1024*1024)
        # return round(fsize,20)
        return fsize

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        # 单线程
        # self.convert_one(0)
        
        # for i in range(len(self)):
        #     self.convert_one(i)
        
        # 多线程
        mmcv.track_parallel_progress(self.convert_one, range(len(self)),self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        # 全景分割里有 3 个 tfrecord 文件会出错, 需要单独处理 
        with open('log/filelist/pass_pseg_training_files.txt','r') as file:
            pass_pseg_training_files = file.readlines()
        pathname = self.tfrecord_pathnames[file_idx]
        # pathname = '/hdd/1000w/waymo/waymo_format/training/segment-5458962501360340931_3140_000_3160_000_with_camera_labels.tfrecord'
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None
                    and frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue
            # 全景分割里有 3 个 tfrecord 文件会出错, 需要单独处理 
            # if pathname + '\n' not in pass_pseg_training_files:
            #     self.save_seg(frame, file_idx, frame_idx)
            # 无需处理
            # self.save_image(frame, file_idx, frame_idx)
            # 这一部分的处理逻辑和 caizhongang 有较大区别
            # cai 只处理了 front camera(camera.name == 1), 这里则处理了5个相机
            # self.save_calib(frame, file_idx, frame_idx)
            # 和 caizhongang 的相比, 多了 elongation, mask_indices
            # self.save_lidar(frame, file_idx, frame_idx)
            self.save_depth(frame, file_idx, frame_idx)
            # 无需修改
            # self.save_pose(frame, file_idx, frame_idx)
            # 原版无, 昊哥建议增加
            # self.save_timestamp(frame, file_idx, frame_idx)

            # if not self.test_mode:
            #     # save_track_id 为 false, caizhongang 是 true, 这点和之前不一样, 影响不大
            #     self.save_label(frame, file_idx, frame_idx)


        # 生成已完成导出的文件列表, 根据日期自动保存到文件
        suffix = time.strftime("%y%m%d", time.localtime())
        suffix_num = 1
        for filename in os.listdir('log/filelist/'):
            if filename.find('finished_filelist_') == 0 and filename.split('finished_filelist_')[1].find(suffix) == 0:
                suffix_num += 1
        filename = 'log/filelist/finished_filelist_' + suffix + "_" + str(suffix_num) + '.txt'
        
        with open(filename,'a') as finished_filelist:
            finished_filelist.write(f'{pathname}\n')
            
    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_seg(self, frame, file_idx, frame_idx):
            """Parse and save the images in png format.

            Args:
                frame (:obj:`Frame`): Open dataset frame proto.
                file_idx (int): Current file index.
                frame_idx (int): Current frame index.
            """
            for img in frame.images:
                if img.camera_segmentation_label.panoptic_label:
                    single_anno = {}
                    seg_path = f'{self.seg_save_dir}{str(img.name - 1)}/' + \
                        f'{self.prefix}{str(file_idx).zfill(3)}' + \
                        f'{str(frame_idx).zfill(3)}.png'
                    img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                        f'{self.prefix}{str(file_idx).zfill(3)}' + \
                        f'{str(frame_idx).zfill(3)}.png'
                    
                    imgname = img_path.split('kitti_format/')[1]
                    segname = seg_path.split('kitti_format/')[1]
                    single_anno['imgname'] = imgname
                    single_anno['segname'] = segname
                    single_seg_anno = []

                    segmentation_proto = img.camera_segmentation_label
                    panoptic_label = decode_single_panoptic_label_from_proto(segmentation_proto)
                    # sementic_label, instance_label = camera_segmentation_utils.decode_semantic_and_instance_labels_from_panoptic_label(
                    #     panoptic_label,
                    #     segmentation_proto.panoptic_label_divisor
                    # )

                    panoptic_label = panoptic_label[:,:,0]

                    H, W = panoptic_label.shape
                    segmentimage = np.zeros((H,W,3))
                    bkmask = np.ones((H,W))

                    color_box = []

                    panoptic_instances = list(np.unique(panoptic_label))
                    for category_id in panoptic_instances:
                        if (category_id // segmentation_proto.panoptic_label_divisor) == cs_pb2.CameraSegmentation.TYPE_UNDEFINED:
                            continue
                        color = random.choice(PALETTE)
                        while color in color_box:
                            color = random.choice(PALETTE)
                        color_box.append(color)
                        mask = (panoptic_label == category_id)
                        bkmask[mask==1] = 0
                        segmentimage[mask] = color

                        B = color[0]
                        G = color[1]
                        R = color[2]

                        single_seg_anno.append(
                            dict(
                                category_id = category_id // segmentation_proto.panoptic_label_divisor ,
                                id = R+G*256+B*256*256
                            )
                        )
                    color = random.choice(PALETTE)
                    while color in color_box:
                        color = random.choice(PALETTE)
                    color_box.append(color)
                    segmentimage[bkmask==1] = color
                    B = color[0]
                    G = color[1]
                    R = color[2]

                    single_seg_anno.append(
                        dict(
                            category_id = cs_pb2.CameraSegmentation.TYPE_UNDEFINED ,
                            id = R+G*256+B*256*256
                        )
                    )
                    single_anno['segmentation'] = single_seg_anno
                    single_anno_json_dir = f'{self.seg_json_save_dir}{str(img.name - 1)}/' + \
                        f'{self.prefix}{str(file_idx).zfill(3)}' + \
                        f'{str(frame_idx).zfill(3)}.json'
                    mmcv.dump(single_anno, single_anno_json_dir)
                    mmcv.imwrite(segmentimage, seg_path)
            
    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in png format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.

        无需处理
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.png'
            img = mmcv.imfrombytes(img.image)
            mmcv.imwrite(img, img_path)

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.

        这一部分的处理逻辑和 caizhongang 有较大区别
        cai 只处理了 front camera(camera.name == 1), 这里则处理了5个相机
        """
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        camera_calibs = []
        R0_rect = [f'{i:e}' for i in np.eye(3).flatten()]
        Tr_velo_to_cams = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(
                4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            if camera.name == 1:  # FRONT = 1, see dataset.proto for details
                self.T_velo_to_front_cam = Tr_velo_to_cam.copy()
            Tr_velo_to_cam = Tr_velo_to_cam[:3, :].reshape((12, ))
            Tr_velo_to_cams.append([f'{i:e}' for i in Tr_velo_to_cam])

            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        # all camera ids are saved as id-1 in the result because
        # camera 0 is unknown in the proto
        # 此处和 caizhongang 有不同之处
        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'
        calib_context += 'R0_rect' + ': ' + ' '.join(R0_rect) + '\n'
        for i in range(5):
            calib_context += 'Tr_velo_to_cam_' + str(i) + ': ' + \
                ' '.join(Tr_velo_to_cams[i]) + '\n'

        with open(
                f'{self.calib_save_dir}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def save_lidar(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.

        和 caizhongang 相比, 多了 elongation, mask_indices
        """
        range_images, camera_projections, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # First return
        # 3d points in vehicle frame: [N, 3] 
        # camera projections of points: [N, 6] 点云在投射到像素平面后的 u v 坐标
        #   包含 2 个 return 的信息, 分别在前 [N,3] 和 后 [N,3]
        # intensity [N, 1] 激光束的返回强度
        # elongation [N, 1] lidar elongation 指脉冲超过其标称宽度的伸长量。例如，具有长脉冲伸长的返回表明激光反射可能会被涂抹或折射，从而使返回脉冲被及时拉长
        # points position in the depth map (element offset if points come from the main lidar otherwise -1) [N, 1]) 
        # All the lists have the length of lidar numbers (5).
        points_0, cp_points_0, intensity_0, elongation_0, mask_indices_0 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        points_0 = np.concatenate(points_0, axis=0)
        intensity_0 = np.concatenate(intensity_0, axis=0)
        elongation_0 = np.concatenate(elongation_0, axis=0)
        mask_indices_0 = np.concatenate(mask_indices_0, axis=0)

        # Second return
        points_1, cp_points_1, intensity_1, elongation_1, mask_indices_1 = \
            self.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=1
            )
        points_1 = np.concatenate(points_1, axis=0)
        intensity_1 = np.concatenate(intensity_1, axis=0)
        elongation_1 = np.concatenate(elongation_1, axis=0)
        mask_indices_1 = np.concatenate(mask_indices_1, axis=0)

        points = np.concatenate([points_0, points_1], axis=0)
        intensity = np.concatenate([intensity_0, intensity_1], axis=0)
        elongation = np.concatenate([elongation_0, elongation_1], axis=0)
        mask_indices = np.concatenate([mask_indices_0, mask_indices_1], axis=0)

        # timestamp = frame.timestamp_micros * np.ones_like(intensity)

        # concatenate x,y,z, intensity, elongation, mask_indices (6-dim)
        point_cloud = np.column_stack(
            (points, intensity, elongation, mask_indices))

        pc_path = f'{self.point_cloud_save_dir}/{self.prefix}' + \
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
        point_cloud.astype(np.float32).tofile(pc_path)

    def save_depth(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, range_image_top_pose = \
            frame_utils.parse_range_image_and_camera_projection(frame)

        # First return
        # points_0: lidar 坐标系下的点云 (x,y,z)
        # cp_points_0: 点云投影到像素平面后在像素坐标系上的 (u,v)
        # 两者都包含 5 个视角的数据
        points_0, cp_points_0 = \
            frame_utils.convert_range_image_to_point_cloud(
                frame,
                range_images,
                camera_projections,
                range_image_top_pose,
                ri_index=0
            )
        # 将 5 个视角的数据合并到一起
        points_0_concat = np.concatenate(points_0, axis=0)
        cp_points_0_concat = np.concatenate(cp_points_0, axis=0)
        
        # 可视化 lidar 点云
        # filename = f'{str(self.prefix)}{str(file_idx).zfill(3)}' +  f'{str(frame_idx).zfill(3)}' 
        # os.makedirs(f'visualized_results/Lidar_Coordinate_Points_0', exist_ok=True)
        # create_output(points_0_concat, np.ones_like(points_0_concat) * 255, f'visualized_results/Lidar_Coordinate_Points_0/{filename}.ply')
        
        # transform_mats: 保存 lidar 坐标系到相机坐标系的转换矩阵
        transform_mats = []
        # waymo front camera to kitti reference camera
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], 
                                       [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])
        for i, camera in enumerate(frame.context.camera_calibrations):
            # extrinsic parameters
            T_cam_to_vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)
            T_vehicle_to_cam = np.linalg.inv(T_cam_to_vehicle)
            Tr_velo_to_cam = \
                self.cart_to_homo(T_front_cam_to_ref) @ T_vehicle_to_cam
            transform_mats.append(Tr_velo_to_cam)

        # points_all: lidar 坐标系下的点云 (x,y,z) 坐标
        # cp_points_all: 点云投影到像素平面后在像素坐标系上的 (u,v)
        points_all = points_0_concat
        cp_points_all = cp_points_0_concat
        
        # 转成 tentor 形式
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)
        points_all_tensor = tf.constant(points_all, dtype=tf.float32)

        images = sorted(frame.images, key=lambda i:i.name)
        for i,image in enumerate(images):
            
            # 当前相机视角下各个数据的 index
            mask = tf.equal(cp_points_all_tensor[..., 0], image.name)
            # 像素坐标系 (u,v)
            cp_points_all_tensor_in_loop = \
                tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
            # 投影到当前相机视角下的点云集合
            points_all_tensor_in_loop = \
                tf.cast(tf.gather_nd(points_all_tensor, tf.where(mask)), dtype=tf.float32)
            # 点云的齐次坐标, 以进行 lidar 坐标系到相机坐标系的转换
            homo_points_all = \
                np.hstack([points_all_tensor_in_loop.numpy(), np.ones((points_all_tensor_in_loop.numpy().shape[0], 1))])
            # lidar 坐标系转相机坐标系后得到相机坐标系下的 (x,y,z)
            # lidar 坐标系转相机坐标系 -> z
            z_depth_all_tensor_in_loop = \
                tf.constant(np.expand_dims((transform_mats[i] @ homo_points_all.T).T[:,-2], axis=1), dtype=tf.float32)
            # lidar 坐标系转相机坐标系, 相机坐标系下的 (x,y)
            xy_all_tensor_in_loop = tf.constant((transform_mats[i] @ homo_points_all.T).T[:,:2], dtype=tf.float32)
            # 组合像素坐标系 (u,v) + z depth 
            uvz = tf.concat(
                [cp_points_all_tensor_in_loop[..., 1:3], z_depth_all_tensor_in_loop], axis=-1).numpy()

            # 当前视角下来自其他 lidar 的点云
            mask2 = tf.equal(cp_points_all_tensor[..., 3], image.name)
            if mask2.shape[0] > 0:
                cp_points_all_tensor_in_loop2 = tf.cast(tf.gather_nd(
                    cp_points_all_tensor, tf.where(mask2)), dtype=tf.float32)
                points_all_tensor_in_loop = \
                    tf.cast(tf.gather_nd(points_all_tensor, tf.where(mask2)), dtype=tf.float32)
                homo_points_all = \
                    np.hstack([points_all_tensor_in_loop.numpy(), np.ones((points_all_tensor_in_loop.numpy().shape[0], 1))])
                z_depth_all_tensor_in_loop2 = \
                    tf.constant(np.expand_dims((transform_mats[i] @ homo_points_all.T).T[:,-2], axis=1), dtype=tf.float32)
                xy_all_tensor_in_loop2 = \
                    tf.constant((transform_mats[i] @ homo_points_all.T).T[:,:2], dtype=tf.float32)
                uvz2 = tf.concat(
                    [cp_points_all_tensor_in_loop2[..., 4:6], z_depth_all_tensor_in_loop2], axis=-1).numpy()
                xy_all_tensor_in_loop = np.concatenate((xy_all_tensor_in_loop, xy_all_tensor_in_loop2), axis = 0)
                uvz = np.concatenate((uvz, uvz2), axis = 0)

            img = mmcv.imfrombytes(image.image)
            xy_coords = uvz[:, :2].astype(np.int32)
            depth_sparse = uvz[:, 2]
            # 全 0 背景 + 点云深度 + scale 200 倍 + 300 米以上做截断
            h, w, _ = img.shape
            depth = np.zeros((h, w))
            depth[xy_coords[:, 1], xy_coords[:, 0]] = depth_sparse
            depth *= 200.
            depth[np.where(depth > 200. * 300.)]= 0
            
            depth_path = f'{self.depth_save_dir}{str(image.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.png'
            mmcv.imwrite(depth.astype(np.uint16), depth_path)

            # # 以下是可视化查看结果
            # rows = xy_coords[...,1]
            # cols = xy_coords[...,0]
            # colors = cm.jet(depth_sparse % 40 / 40.0)
            # fig, ax = plt.subplots(1, 1, figsize=(20, 20))
            # fig.tight_layout()
            # undist_imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 数据集 rgb 图片
            # ax.axis('off')
            # ax.imshow(undist_imgrgb)
            # ax.scatter(cols, rows, c=colors, s=3)

            # filename = f'{str(self.prefix)}{str(file_idx).zfill(3)}' + \
            #             f'{str(frame_idx).zfill(3)}_{image.name}' 
            
            # # os.makedirs(f'visualized_results/Scat_Points/proj_once', exist_ok=True)
            # # plt.savefig(f'visualized_results/Scat_Points/proj_once/{filename}.png', bbox_inches='tight',pad_inches = 0)
            # os.makedirs(f'visualized_results/Scat_Points/proj_twice', exist_ok=True)
            # plt.savefig(f'visualized_results/Scat_Points/proj_twice/{filename}.png', bbox_inches='tight',pad_inches = 0)

            # points = np.dstack((xy_coords[:,0], xy_coords[:,1], depth_sparse*100)) 
            # points = np.squeeze(points)
            # os.makedirs(f'visualized_results/Camera_Coordinate_Points/proj_once', exist_ok=True)
            # create_output(points, np.ones_like(points) * 255, f'visualized_results/Camera_Coordinate_Points/proj_once/{filename}.ply')
            # os.makedirs(f'visualized_results/Camera_Coordinate_Points/proj_twice', exist_ok=True)
            # create_output(points, np.ones_like(points) * 255, f'visualized_results/Camera_Coordinate_Points/proj_twice/{filename}.ply')
    
            
    
    def save_label(self, frame, file_idx, frame_idx):
        """Parse and save the label data in txt format.
        The relation between waymo and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (waymo) -> l, h, w (kitti)
        2. x-y-z: front-left-up (waymo) -> right-down-front(kitti)
        3. bbox origin at volumetric center (waymo) -> bottom center (kitti)
        4. rotation: +x around y-axis (kitti) -> +x around z-axis (waymo)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        save_track_id 为 false, caizhongang 是 true, 这点和之前不一样, 影响不大
        """
        fp_label_all = open(
            f'{self.label_all_save_dir}/{self.prefix}' +
            f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'w+')
        id_to_bbox = dict()
        id_to_name = dict()
        for labels in frame.projected_lidar_labels:
            name = labels.name
            for label in labels.labels:
                # TODO: need a workaround as bbox may not belong to front cam
                bbox = [
                    label.box.center_x - label.box.length / 2,
                    label.box.center_y - label.box.width / 2,
                    label.box.center_x + label.box.length / 2,
                    label.box.center_y + label.box.width / 2
                ]
                id_to_bbox[label.id] = bbox
                id_to_name[label.id] = name - 1

        for obj in frame.laser_labels:
            bounding_box = None
            name = None
            id = obj.id
            for lidar in self.lidar_list:
                if id + lidar in id_to_bbox:
                    bounding_box = id_to_bbox.get(id + lidar)
                    name = str(id_to_name.get(id + lidar))
                    break

            if bounding_box is None or name is None:
                name = '0'
                bounding_box = (0, 0, 0, 0)

            my_type = self.type_list[obj.type]

            if my_type not in self.selected_waymo_classes:
                continue

            if self.filter_empty_3dboxes and obj.num_lidar_points_in_box < 1:
                continue

            my_type = self.waymo_to_kitti_class_map[my_type]

            height = obj.box.height
            width = obj.box.width
            length = obj.box.length

            x = obj.box.center_x
            y = obj.box.center_y
            z = obj.box.center_z - height / 2

            # project bounding box to the virtual reference frame
            pt_ref = self.T_velo_to_front_cam @ \
                np.array([x, y, z, 1]).reshape((4, 1))
            x, y, z, _ = pt_ref.flatten().tolist()

            rotation_y = -obj.box.heading - np.pi / 2
            track_id = obj.id

            # not available
            truncated = 0
            occluded = 0
            alpha = -10

            line = my_type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            if self.save_track_id:
                line_all = line[:-1] + ' ' + name + ' ' + track_id + '\n'
            else:
                line_all = line[:-1] + ' ' + name + '\n'

            fp_label = open(
                f'{self.label_save_dir}{name}/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

            fp_label_all.write(line_all)

        fp_label_all.close()

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        无需修改
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/{self.prefix}' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def save_timestamp(self, frame, file_idx, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        
        cai 无
        """
        with open(
                join(f'{self.timestamp_save_dir}/{self.prefix}' +
                     f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
                'w') as f:
            f.write(str(frame.timestamp_micros))

    def create_folder(self):
        """Create folder for data preprocessing."""
        # create all folders
        if not self.test_mode:
            dir_list1 = [
                self.label_all_save_dir, self.calib_save_dir,
                self.point_cloud_save_dir, self.pose_save_dir,
                self.timestamp_save_dir
            ]
            dir_list2 = [
                self.label_save_dir, self.image_save_dir, 
                self.depth_save_dir, self.seg_save_dir,
                self.seg_json_save_dir
            ]
        else:
            dir_list1 = [
                self.calib_save_dir, self.point_cloud_save_dir,
                self.pose_save_dir, self.timestamp_save_dir
            ]
            dir_list2 = [self.image_save_dir, self.depth_save_dir, self.seg_save_dir, self.seg_json_save_dir]
        for d in dir_list1:
            mmcv.mkdir_or_exist(d)
        for d in dir_list2:
            for i in range(5):
                mmcv.mkdir_or_exist(f'{d}{str(i)}')

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(
            frame.context.laser_calibrations, key=lambda c: c.name)
        points = []
        cp_points = []
        intensity = []
        elongation = []
        mask_indices = []

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            # c.name = 1 2 3 4 5, 表示 5 个 lidar 的序号
            # range_image: 由第 c.name 号 lidar 的第 ri_index 次 return 的点云编码得到
            # shape [X,Y,4]
            #   channel 0: range
            #   channel 1: lidar intensity
            #   channel 2: lidar elongation
            #   channel 3: is_in_nlz (1 = in, -1 = not in)
            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            # range_image_tensor 中 range 数值大于 0 的表示有效的点云位置          
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask
            # range image 转为笛卡尔坐标系下的坐标
            # (range, azimuth, inclination) -> (x,y,z)
            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)
            # 有效点云位置
            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian, mask_index)

            # camera_projections: [5, 2, N, 6]
            # cp: [N,6]
            # cp: 由第 c.name 号 lidar 的第 ri_index 次 return 的点云投影到相机平面
            cp = camera_projections[c.name][ri_index]
            cp_tensor = tf.reshape(
                tf.convert_to_tensor(value=cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, mask_index)
            points.append(points_tensor.numpy())
            cp_points.append(cp_points_tensor.numpy())

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)
            intensity.append(intensity_tensor.numpy())

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)
            elongation.append(elongation_tensor.numpy())
            if c.name == 1:
                mask_index = (ri_index * range_image_mask.shape[0] +
                              mask_index[:, 0]
                              ) * range_image_mask.shape[1] + mask_index[:, 1]
                mask_index = mask_index.numpy().astype(elongation[-1].dtype)
            else:
                mask_index = np.full_like(elongation[-1], -1)

            mask_indices.append(mask_index)

        return points, cp_points, intensity, elongation, mask_indices

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret
    
    def rgba(self, r):
        """Generates a color based on range.

        Args:
            r: the range value of a given point.
        Returns:
            The color for a given range
        """
        c = plt.get_cmap('jet')((r % 20.0) / 20.0)
        c = list(c)
        c[-1] = 0.5  # alpha
        return c

    def plot_image(self, camera_image):
        """Plot a cmaera image."""
        plt.figure(figsize=(20, 12))
        plt.imshow(tf.image.decode_jpeg(camera_image.image))
        plt.grid("off")

    def plot_points_on_image(self, projected_points, camera_image, depth_map_path, rgba_func,
                            point_size=5.0):
        """Plots points on a camera image.

        Args:
            projected_points: [N, 3] numpy array. The inner dims are
            [camera_x, camera_y, range].
            camera_image: jpeg encoded camera image.
            rgba_func: a function that generates a color from a range value.
            point_size: the point size.

        """
        self.plot_image(camera_image)

        xs = []
        ys = []
        colors = []

        for point in projected_points:
            xs.append(point[0])  # width, col
            ys.append(point[1])  # height, row
            colors.append(rgba_func(point[2]))

        plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

        depth_map_path = depth_map_path.split('.png')[0]+'_plt.png'
        plt.savefig(depth_map_path)