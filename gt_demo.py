import numpy as np
from PIL import Image
import PIL
import cv2
import pybullet as p
import os
from scipy.spatial.transform import Rotation as Rot
from utils import visualization_parts
import time
import argparse
# automatically label the urdf based on the masks
# labeling the parent:
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.configureDebugVisualizer(1, lightPosition=(1250, 100, 2000), rgbBackground=(1, 1, 1))

def visualize_gt_global(test_id, scene_name, if_random, with_texture):
    p.resetSimulation()

    num_roots_global = 5
    data_path = f"assets/{scene_name}/labels/label{test_id}.npy"
    data = np.load(data_path, allow_pickle=True).item()


    mesh_pred_global = data['meshes']
    extended_position_pred= data['positions_start']
    extended_position_end= data['positions_end']
    parent_pred_global= data['relations']
    extended_position_pred_new = np.zeros((len(extended_position_pred), 3))
    extended_position_pred_new[:, 1:] = extended_position_pred

    extended_position_end_new = np.zeros((len(extended_position_end), 3))
    extended_position_end_new[:, 1:] = extended_position_end

    scale_pred_global = abs(np.array((extended_position_end_new - extended_position_pred_new)))
    scale_pred_global[:, 0] = 1

    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[5 + mesh_id]),
                                       parent_pred_global[5 + mesh_id].shape)
        parent_id = each_parent[0]

        if parent_id!=2:

            extended_position_pred_new[mesh_id] = np.array([extended_position_end[mesh_id][0], 0.0, extended_position_pred[mesh_id][1]])
            extended_position_end_new[mesh_id] = [extended_position_end[mesh_id][0], 1.0, extended_position_pred[mesh_id][1]]
            scale_pred_global[mesh_id] = [(extended_position_end[mesh_id][0] - extended_position_pred[mesh_id][0]), 1, extended_position_end[mesh_id][1] - extended_position_pred[mesh_id][1]]

    front_object_position_end = []

    mesh_pred_parts = data['part_meshes']
    part_positions_starts = data['part_positions_start']
    part_positions_ends = data['part_positions_end']
    part_relations = data['part_relations']
    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]), parent_pred_global[num_roots_global + mesh_id].shape)

        parent_id = each_parent[0]
        if parent_id==4:
            continue

        position_pred_part = part_positions_starts[mesh_id]
        position_pred_end_part = part_positions_ends[mesh_id]
        mesh_pred_part = mesh_pred_parts[mesh_id]
        parent_pred_part = part_relations[mesh_id]
        base_pred = int(data['part_bases'][mesh_id])


        if parent_id<=2:

            root_position = extended_position_pred_new[mesh_id] + np.array([1.1, 0.1, 0])
            root_orientation = [0, 0, 0, 1]
            root_scale = scale_pred_global[mesh_id]

            root_scale[1] = root_scale[1] - 0.05 # making a little gap between two objects
        elif parent_id==3:
            root_orientation = Rot.from_rotvec([0,0,np.pi/2]).as_quat()
            root_scale = [scale_pred_global[mesh_id][1], scale_pred_global[mesh_id][0], scale_pred_global[mesh_id][2]]
            root_position = extended_position_pred_new[mesh_id] + np.array([0, 1.1, 0])


        root_scale[2] = root_scale[2] - 0.2

        size_scale = 4
        scale_pred_part = abs(np.array(size_scale * (np.array(position_pred_end_part) - np.array(position_pred_part))/ 12))
        scale_pred_part[:, 1] *= root_scale[1]
        scale_pred_part[:, 2] *= root_scale[2]



        if base_pred==4:
            root_position[2]=0.4*root_scale[2]


        texture_list = []
        if with_texture:
            ############## load texture if needed ##################
            label_path = f'assets/{scene_name}/labels/label{test_id}.npy'
            object_info = np.load(label_path, allow_pickle=True).item()
            bboxes = object_info['part_normalized_bbox'][mesh_id]

            for bbox_id in range(len(bboxes)):
                if os.path.exists(f"assets/{scene_name}/textures/test{test_id}/{mesh_id}/{bbox_id}.png"):
                    texture_list.append(f"assets/{scene_name}/textures/test{test_id}/{mesh_id}/{bbox_id}.png")
                else:
                    print('no texture map found! Run get_texture.py first')

        visualization_parts(p, root_position, root_orientation, root_scale,  base_pred, np.array(position_pred_part).astype(int), scale_pred_part, mesh_pred_part, parent_pred_part, texture_list, if_random, filename=f"assets/{scene_name}/urdfs/{test_id}")

        if parent_id<=2:
            front_object_position_end.append(extended_position_end_new[mesh_id][1])
    for mesh_id, each_mesh in enumerate(mesh_pred_global):
        each_parent = np.unravel_index(np.argmax(parent_pred_global[num_roots_global + mesh_id]),
                                       parent_pred_global[num_roots_global + mesh_id].shape)

        parent_id = each_parent[0]
        if parent_id == 4:
            right_wall_distance = max(front_object_position_end)
            # get the cropped image


            position_pred_part = part_positions_starts[mesh_id]
            position_pred_end_part = part_positions_ends[mesh_id]
            mesh_pred_part = mesh_pred_parts[mesh_id]
            parent_pred_part = part_relations[mesh_id]
            base_pred = int(data['part_bases'][mesh_id])

            root_orientation = Rot.from_rotvec([0, 0, -np.pi / 2]).as_quat()
            root_scale = [scale_pred_global[mesh_id][1], scale_pred_global[mesh_id][0],
                          scale_pred_global[mesh_id][2]]

            root_position = extended_position_pred_new[mesh_id] + np.array(
                [-scale_pred_global[mesh_id][0], right_wall_distance, 0])
            root_scale[2] = root_scale[2] - 0.2

            size_scale = 4
            scale_pred_part = abs(np.array(size_scale * (position_pred_end_part - position_pred_part) / 12))
            scale_pred_part[:, 1] *= root_scale[1]
            scale_pred_part[:, 2] *= root_scale[2]
            if base_pred == 4:
                root_position[2] = 0.4 * root_scale[2]

            texture_list = []
            if with_texture:
                ############## load texture if needed ##################
                label_path = f'assets/{scene_name}/labels/label{test_id}.npy'
                object_info = np.load(label_path, allow_pickle=True).item()
                bboxes = object_info['part_normalized_bbox'][mesh_id]

                for bbox_id in range(len(bboxes)):
                    if os.path.exists(f"assets/{scene_name}/textures/test{test_id}/{mesh_id}/{bbox_id}.png"):
                        texture_list.append(f"assets/{scene_name}/textures/test{test_id}/{mesh_id}/{bbox_id}.png")
                    else:
                        print('no texture map found! Run get_texture.py first')

            visualization_parts(p, root_position, root_orientation, root_scale, base_pred,
                                np.array(position_pred_part).astype(int), scale_pred_part, mesh_pred_part,
                                parent_pred_part, texture_list, if_random,
                                filename=f"assets/{scene_name}/urdfs/{test_id}")

def visualize_gt_obj(test_id, scene_name, if_random, with_texture):

    p.resetSimulation()
    data = np.load(f'assets/{scene_name}/labels/label{test_id}.npy', allow_pickle=True).item()
    position_start = data['part_positions_start']
    position_end = data['part_positions_end']
    parent_labels = data['part_relations']
    mesh_types = data['part_meshes']
    base_pred = data['part_bases']

    extended_position_pred = np.zeros((len(position_start), 3))
    extended_position_pred[:, 1:] = position_start

    extended_position_end_pred = np.zeros((len(position_end), 3))
    extended_position_end_pred[:, 1:] = position_end

    extended_scale_pred = np.ones((len(position_end), 3))
    extended_scale_pred[:, 1:] = abs(np.array(4 * (position_end - position_start) / 12))  # ((end-start)/bin_unit)/0.25

    size_scale = 4
    scale_pred_part = abs(np.array(size_scale * (np.array(position_end) - np.array(position_start)) / 12))

    # fifth, sanity check.
    root_position = [0, 0, 0]
    root_orientation = [0, 0, 0, 1]

    root_scale = [1, 1, 1]
    if scene_name=="fridges":
        root_scale[2] =2
    extended_scale_pred[:, 1] *= root_scale[1]
    extended_scale_pred[:, 2] *= root_scale[2]


    texture_list = []
    if with_texture:
        ############## load texture if needed ##################
        label_path = f'assets/{scene_name}/labels/label{test_id}.npy'
        object_info = np.load(label_path, allow_pickle=True).item()
        bboxes = object_info['part_normalized_bbox']

        for bbox_id in range(len(bboxes)):
            if os.path.exists(f"assets/{scene_name}/textures/test{test_id}/{bbox_id}.png"):
                texture_list.append(f"assets/{scene_name}/textures/test{test_id}/{bbox_id}.png")
            else:
                print('no texture map found! Run get_texture.py first')

    visualization_parts(p, root_position, root_orientation, root_scale, base_pred, extended_position_pred, extended_scale_pred, mesh_types, parent_labels, texture_list, if_random, filename=f"assets/{scene_name}/urdfs/{test_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--texture', action='store_true', help='adding texture')
    parser.add_argument('--random', action='store_true', help='adding texture')
    parser.add_argument('--scene', '--scene', default='objects', type=str)
    args = parser.parse_args()

    if_random = args.random
    with_texture = args.texture
    if args.scene == "objects":
        data_sizes = [100, 50, 50, 50, 50]
        for scene_id, scene_name in enumerate(['cabinets', 'ovens', 'fridges', 'dishwashers', 'washers']):
            for test_id in range(data_sizes[scene_id]):
                image_path = f"assets/{scene_name}/images/test{test_id}.jpg"
                image_obj = Image.open(image_path).show()
                visualize_gt_obj(test_id, scene_name, if_random, with_texture)
                time.sleep(1)
    elif args.scene == "kitchens":
        for test_id in range(54):
            image_path = f"assets/{args.scene}/images/test{test_id}.jpg"
            image_obj = Image.open(image_path).show()

            visualize_gt_global(test_id, args.scene, if_random, with_texture)
            time.sleep(1)
    else:
        print("use the valid name to define the scene. choose from ['kitchens', 'objects']")
