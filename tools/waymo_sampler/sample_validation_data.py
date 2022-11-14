import cv2
from os import path as osp
import random
import os
import numpy as np
import json
def create_filepaths(source_root):
    files = []
    for file in os.listdir(osp.join(source_root, 'depthmap_0/')):
        files.append(file)
    slice = random.sample(files, 2000)
    return slice

def get_filenames(source_root):
    for _,_,files in os.walk(source_root):
        filenames = files
    return filenames

def sample_validation_data(source_root, des_root):
    """
        source_root = /hdd3/1000w/waymo/kitti_format/validation/
        des_root = /hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/
        '/hdd3/1000w/waymo/sampled_validation/': 表示工作目录
        'waymo...': 实际上从这里开始打包

        waymo
        ├─ kitti_format
        │   ├─ validation
        │   │   ├─ depthmap_0
        │   │   ├─ depthmap_1
        │   │   ├─ depthmap_2
        │   │   ├─ depthmap_3
        │   │   ├─ depthmap_4
        │   │   ├─ image_0
        │   │   ├─ image_1
        │   │   ├─ image_2
        │   │   ├─ image_3
        │   │   ├─ image_4
        │   │   ├─ valid_annotation.json
        │   │   ├─ intrinsic.json
        
        注意 sample 数据集和 json 需要同时生成, 不能分开做, 否则 json 文件匹配不上
    """
        
    # 预先创建文件夹
    for camera_index in range(5):
        if not osp.exists(osp.join(des_root, f'depthmap_{str(camera_index)}')):
            os.makedirs(osp.join(des_root, f'depthmap_{str(camera_index)}'))
        if not osp.exists(osp.join(des_root, f'image_{str(camera_index)}')):
            os.makedirs(osp.join(des_root, f'image_{str(camera_index)}'))

    # filename: list ['1072002.png', '1119123.png', ...]
    # 得到 2000 帧图片的名字
    filenames = create_filepaths(source_root)

    camara_list = [ 
            'front', 'front_left', 
            'front_right', 'side_left',
            'side_right']

    # 处理 2000 帧
    anno_dict_list = []
    for filename in filenames:

        # 处理 5 个相机的深度图和 rgb 图
        anno_dict = {}
        for camera_index in range(5):
            
            depthmap_source_path = osp.join(source_root, f'depthmap_{str(camera_index)}', filename)
            depthmap_des_path = osp.join(des_root, f'depthmap_{str(camera_index)}', filename)
            
            # os.system(f'cp {depthmap_source_path} {depthmap_des_path}')

            rgb_source_path = osp.join(source_root, f'image_{str(camera_index)}', filename)
            rgb_des_path = osp.join(des_root, f'image_{str(camera_index)}', filename)
            # os.system(f'cp {rgb_source_path} {rgb_des_path}')

            anno_dict[f'rgb_{camara_list[camera_index]}_path'] = f'{rgb_des_path.split("sampled_validation/")[1]}'
            anno_dict[f'depth_{camara_list[camera_index]}_path'] = f'{depthmap_des_path.split("sampled_validation/")[1]}'
            
            intrinsic_index = [1,3,6,7] 
            calib_source_path = osp.join(source_root, 'calib', f'{filename.split(".")[0]}.txt')
            with open(calib_source_path, 'r') as calib:
                intrinsics = calib.readlines()
                anno_dict[f'intrinsic_{camara_list[camera_index]}'] = [intrinsics[camera_index].split(' ')[i] for i in intrinsic_index]

        anno_dict_list.append(anno_dict)

    with open('/hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/valid_annotations_2.json', 'w') as outfile:
        json.dump(anno_dict_list, outfile, indent = 4)


def test_annotation_json(anno_json_path, workspace_root):
    
    annos = json.load(open(anno_json_path, 'r'))
    paths = []
    
    for anno in annos:
        paths.append(osp.join(workspace_root, anno['rgb_front_path']))
        paths.append(osp.join(workspace_root, anno['rgb_front_left_path']))
        paths.append(osp.join(workspace_root, anno['rgb_front_right_path']))
        paths.append(osp.join(workspace_root, anno['rgb_side_left_path']))
        paths.append(osp.join(workspace_root, anno['rgb_side_right_path']))
        paths.append(osp.join(workspace_root, anno['depth_front_path']))
        paths.append(osp.join(workspace_root, anno['depth_front_left_path']))
        paths.append(osp.join(workspace_root, anno['depth_front_right_path']))
        paths.append(osp.join(workspace_root, anno['depth_side_left_path']))
        paths.append(osp.join(workspace_root, anno['depth_side_right_path']))
    imgnum = 0
    for path in paths:
        if os.path.exists(path):
            imgnum +=1 
        
    print(imgnum)

def generate_validation_json(root_path):
    """
        root_path = /hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/
        '/hdd3/1000w/waymo/sampled_validation/': 表示工作目录
        'waymo': 实际上从这里开始打包

        waymo
        ├─ kitti_format
        │   ├─ validation
        │   │   ├─ depthmap_0
        │   │   ├─ depthmap_1
        │   │   ├─ depthmap_2
        │   │   ├─ depthmap_3
        │   │   ├─ depthmap_4
        │   │   ├─ image_0
        │   │   ├─ image_1
        │   │   ├─ image_2
        │   │   ├─ image_3
        │   │   ├─ image_4
        │   │   ├─ valid_annotation.json
        │   │   ├─ intrinsic.json
        
        注意 sample 数据集和 json 需要同时生成, 不能分开做, 否则 json 文件匹配不上
    """
        
    # filename: list ['1072002.png', '1119123.png', ...]
    # 得到 2000 帧图片的名字
    filenames = get_filenames(osp.join(root_path, 'depthmap_2'))

    camara_list = [ 
            'front', 'front_left', 
            'front_right', 'side_left',
            'side_right']

    # 处理 2000 帧
    anno_dict_list = []
    for filename in filenames:

        # 处理 5 个相机的深度图和 rgb 图
        anno_dict = {}
        for camera_index in range(5):
            
            rgb_path = osp.join(f'waymo/kitti_format/validation/image_{str(camera_index)}', filename)
            depthmap_path = osp.join(f'waymo/kitti_format/validation/depthmap_{str(camera_index)}', filename)

            anno_dict[f'rgb_{camara_list[camera_index]}_path'] = rgb_path
            anno_dict[f'depth_{camara_list[camera_index]}_path'] = depthmap_path
            
            intrinsic_index = [1,3,6,7] 
            calib_source_path = osp.join('/hdd3/1000w/waymo/kitti_format/validation/calib', f'{filename.split(".")[0]}.txt')
            with open(calib_source_path, 'r') as calib:
                intrinsics = calib.readlines()
                anno_dict[f'intrinsic_{camara_list[camera_index]}'] = [intrinsics[camera_index].split(' ')[i] for i in intrinsic_index]

        anno_dict_list.append(anno_dict)

    with open('/hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/valid_annotations_2.json', 'w') as outfile:
        json.dump(anno_dict_list, outfile, indent = 4)


# sample_validation_data('/hdd3/1000w/waymo/kitti_format/validation/', '/hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/')

# test_annotation_json('/hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/valid_annotations.json', '/hdd3/1000w/waymo/sampled_validation')


generate_validation_json('/hdd3/1000w/waymo/sampled_validation/waymo/kitti_format/validation/')