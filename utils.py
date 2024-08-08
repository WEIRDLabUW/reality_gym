import pybullet as p
import torch
import random
import glob
import os
from scipy.spatial.transform import Rotation as Rot
import numpy as np



def domain_randomization(root_name, link_names, random_frame,  linkJointAxis, jointtypes):
    # ransomize assets
    asset_path = "/home/zoeyc/github/reality_gym/RSS/robots/simulation/partnet_uv"
    # get what's the id for each part.
    new_drawer = random.choice(glob.glob(asset_path+"/drawers/*.obj"))
    new_doorL = random.choice(glob.glob(asset_path + "/doorLs/*.obj"))
    new_doorR = asset_path + "/doorRs/{}".format(os.path.basename(new_doorL))
    # if random_frame:
    root_name =  random.choice(glob.glob(asset_path + "/cabinet_frames/*.obj"))
    # pick a random handle
    new_handle = random.choice(glob.glob(asset_path + "/{}/*.obj".format(random.choice(['handles', 'knobs']))))

    new_links = link_names.copy()

    for link_id, each_name in enumerate(link_names):
        # if random_frame:
        if "drawer" in each_name:
            new_links[link_id] = new_drawer
        if "doorL" in each_name:
            new_links[link_id] = new_doorL
            # swap = new_drawer  # random.choice([new_doorL, new_drawer])
            # new_links[link_id] = swap
            # if new_drawer == swap:
            #     linkJointAxis[link_id] = [1, 0, 0]
            #     jointtypes[link_id] = 1
        if "doorR" in each_name:
            new_links[link_id] = new_doorR
        if "handle" in each_name or "knob" in each_name:
            new_links[link_id] = new_handle
    return root_name, new_links, linkJointAxis, jointtypes


def visual_collision_shapes(links, link_scales, link_positions):
    collisionIndices, visualIndices = [], []

    for i, each_link in enumerate(links):
        if "handle" in each_link or "knob" in each_link:
            color = [0.6,0.6,0.6,1]
            link_scales[i] = [1,0.8,0.8]#[1, 0.1, 3]
            # link_scales[i] = [2, 1, 5]
            # breakpoint()
            # link_positions[i][1]-=np.random.uniform(0.15, 0.16)
        else:
            color = [1,1,1,1]
        # else:
        #     color = [np.random.uniform(0.75, 0.76), np.random.uniform(0.75, 0.76), np.random.uniform(0.75, 0.76),1]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                            fileName=each_link,
                                            rgbaColor=[color[0],color[1],color[2], 1],
                                            specularColor=[0.5, .4, 0],
                                            meshScale=link_scales[i],
                                            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
                                            )

        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                  fileName=each_link,
                                                  meshScale=link_scales[i],
                                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                                  )

        visualIndices.append(visualShapeId)
        collisionIndices.append(collisionShapeId)


    return collisionIndices, visualIndices, link_positions

def create_obj(p, obj_path, scale, obj_t, obj_q):
    base_visualid = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path,
        rgbaColor=None,
        meshScale=list(scale),
    )

    base_collisionid = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=obj_path[:-4]+"_vhacd.obj",
        meshScale=list(scale),
    )

    new_obj = p.createMultiBody(random.uniform(0.05, 0.2),
                                base_collisionid,
                                base_visualid,
                                obj_t,
                                obj_q
                                )

    return new_obj



def visualization_parts(p, root_position, root_orientation, root_scale, mesh_base, position_pred_ori, scale_pred_ori, mesh_pred_ori, parent_pred_ori, texture_list, if_random, filename="output"):
    num_roots = 1

    part_names = ['none', 'drawer', 'doorL', 'doorR',
                  'handle', 'knob', 'washer_door', 'doorD', 'oven_door', 'doorU']
    base_names = ['none', 'cabinet_kitchen', 'oven', 'dishwasher', 'washer', 'fridge',
                  'oven_fan', 'shelf_base'
                  ]
    part_path = "assets/all_meshes/parts/"

    # fix the oven fan problem: if there are parts, root shouldn't be oven fan
    if len(position_pred_ori) > 1 and mesh_base == 6:
        mesh_base = 1


    root = "assets/all_meshes/{}.obj".format(base_names[mesh_base])

    position_type = np.arange(13)/12
    links = []
    link_scales = []
    link_positions = []
    link_orientations = []
    linkparents = []
    jointtypes = []
    linkJointAxis = []

    parent_pred = []
    relations_pred = []


    if mesh_base >=9: # rigid objects
        object_path = "meshes/{}.obj".format(base_names[mesh_base])
        obj = create_obj(p, object_path, root_scale, root_position, root_orientation)
        p.changeVisualShape(obj, -1, rgbaColor=[0.6, 0.6, 0.6, 1])
        # if base_names[mesh_base]=="square_table":
        if mesh_base>9:
            base_texture = random.choice(glob.glob("default_textures/cab_wood/*"))
            base_tex = p.loadTexture(base_texture)
            p.changeVisualShape(obj, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex)
    else:
        for i, each_parent in enumerate(parent_pred_ori[num_roots:]):
            each_parent = np.unravel_index(np.argmax(each_parent), each_parent.shape)
            parent_id = each_parent[0]
            parent_pred.append(parent_id)
            relations_pred.append(each_parent[1])

        parent_pred = np.array(parent_pred)
        mesh_pred = np.array(mesh_pred_ori)
        scale_pred= np.array(scale_pred_ori)
        position_pred = np.array(position_pred_ori)

        new_order = np.argsort(parent_pred)
        parent_pred, scale_pred, mesh_pred, position_pred = parent_pred[new_order], scale_pred[new_order], mesh_pred[new_order], position_pred[new_order]

        if len(texture_list)>0:

            texture_list = [texture_list[i] for i in new_order]
        # update parent labels
        for p_i, each_parent in enumerate(parent_pred):
            if each_parent>0:
                if each_parent-1 in list(new_order):
                    parent_pred[p_i] = list(new_order).index(each_parent-1)+1

        for i, each_parent in enumerate(parent_pred):
            each_mesh = int(mesh_pred[i])
            parent_id = each_parent

            if each_mesh ==1:
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1]-0.02, scale_pred[i][2]-0.02])

                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_PRISMATIC)
                linkJointAxis.append([1,0,0])
            elif each_mesh ==2:
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1]-0.01, scale_pred[i][2]-0.01])

                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0, 0, 0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

            elif each_mesh ==3:
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1]-0.01, scale_pred[i][2]-0.01])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,-1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh ==6: # washerdoor
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])
            elif each_mesh ==7: # down door
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0, 1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])
            elif each_mesh ==8: # oven door:
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0,1, 0])
            elif each_mesh ==9: # up door
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 1, 0])

            elif each_mesh ==4:
                links.append(part_path+part_names[each_mesh]+".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])

                if each_parent-1>=len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue
                if mesh_pred[each_parent-1]==2: # left door
                    link_positions.append([0, 0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent-1]== 3: # right door
                    link_positions.append([0, -0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent-1]==7:
                    link_positions.append(
                        [0, 0.25 * scale_pred[each_parent - 1][1] * position_type[position_pred[i][1].astype(int)],
                         0.25 * scale_pred[each_parent - 1][2] * position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(p.getQuaternionFromEuler([np.pi / 2, 0, 0]))

                elif mesh_pred[each_parent-1]==8:
                    link_positions.append(
                        [0, 0.25 * scale_pred[each_parent - 1][1] * position_type[6],
                         0.25 * scale_pred[each_parent - 1][2] * position_type[10]])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(p.getQuaternionFromEuler([np.pi / 2, 0, 0]))

                elif mesh_pred[each_parent-1]==9:
                    link_positions.append(
                        [0, 0.25 * scale_pred[each_parent - 1][1] * position_type[position_pred[i][1].astype(int)],
                         -0.25 * scale_pred[each_parent - 1][2] * position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(p.getQuaternionFromEuler([np.pi / 2, 0, 0]))


                elif mesh_pred[each_parent-1]== 1:  # drawer
                    link_positions.append([0,  0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
            elif each_mesh ==5:
                links.append(part_path+part_names[each_mesh]+".obj")
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_FIXED)
                link_scales.append([1, 1, 1])
                if each_parent-1>=len(mesh_pred):
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])
                    continue
                if mesh_pred[each_parent-1]==2: # left door
                    link_positions.append([0, 0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent-1] == 3: # right door
                    link_positions.append([0, -0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([1, 0, 0])
                    link_orientations.append([0, 0, 0, 1])
                elif mesh_pred[each_parent-1] == 1:  # drawer
                    link_positions.append([0,  0.25*scale_pred[each_parent-1][1]*position_type[position_pred[i][1].astype(int)], 0.25*scale_pred[each_parent-1][2]*position_type[position_pred[i][2].astype(int)]])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append(p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
                else:
                    link_positions.append([0, 0, 0])
                    linkJointAxis.append([0, 0, 1])
                    link_orientations.append([0, 0, 0, 1])

            elif each_mesh ==6: # washer door
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1].astype(int)]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,-1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

            elif each_mesh ==7: # down door
                links.append(part_path+part_names[each_mesh]+".obj")
                link_scales.append([1, scale_pred[i][1], scale_pred[i][2]])
                link_positions.append([0, position_type[position_pred[i][1]]*root_scale[1], position_type[position_pred[i][2].astype(int)]*root_scale[2]])
                link_orientations.append([0,0,0,-1])
                linkparents.append(parent_id)
                jointtypes.append(p.JOINT_REVOLUTE)
                linkJointAxis.append([0, 0, 1])

        abs_link_scales = [[abs(x) for x in sublist] for sublist in link_scales]
        obj = create_articulated_objects(root, root_scale, root_position, root_orientation, links, abs_link_scales, link_positions, link_orientations, linkparents, jointtypes, linkJointAxis, texture_list, if_random)

    # write_urdfs(filename, root, root_scale, root_position, root_orientation, links, link_scales, link_positions,
    #             link_orientations, linkparents, jointtypes, linkJointAxis)

    return obj, link_orientations

def write_numpy(filename, root, root_scale, root_position, root_orientation, links, link_scales,
                    link_positions, link_orientations, linkparents, jointtypes, linkJointAxis):

    urdf_primitives = {}
    urdf_primitives['root'] = root
    urdf_primitives['root_scale'] = root_scale
    urdf_primitives['root_position'] = root_position
    urdf_primitives['root_orientation'] = root_orientation
    urdf_primitives['links'] = links
    urdf_primitives['link_scales'] = link_scales
    urdf_primitives['link_positions'] = link_positions
    urdf_primitives['link_orientations'] = link_orientations
    urdf_primitives['linkparents'] = linkparents
    urdf_primitives['jointtypes'] = jointtypes
    urdf_primitives['linkJointAxis'] = linkJointAxis

    np.save(filename, urdf_primitives)


def write_urdfs(filename, root, root_scale, root_position, root_orientation, links, link_scales,  link_positions, link_orientations, linkparents, jointtypes, linkJointAxis):
    root_rot = Rot.from_quat(root_orientation).as_rotvec()
    import xml.etree.ElementTree as ET
    import os
    from xml.dom import minidom
    def prettify(elem):
        """Return a pretty-printed XML string for the Element."""
        rough_string = ET.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    # Create the root element of the URDF file, which is <robot>
    joint_names = ['revolute', 'prismatic', 'spherical', 'unknown',  'fixed']
    robot = ET.Element('robot', attrib={'name': 'my_urdf'})

    # Add a base link element with a mesh file for the visual geometry
    base_link = ET.SubElement(robot, 'link', attrib={'name': 'base_link'})
    visual = ET.SubElement(base_link, 'visual')
    ET.SubElement(visual, 'origin', attrib={'xyz': '{0} {1} {2}'.format(root_position[0], root_position[1], root_position[2]), 'rpy': '{0} {1} {2}'.format(root_rot[0], root_rot[1], root_rot[2])})

    material = ET.SubElement(visual, 'material', attrib={'name': 'white'})
    ET.SubElement(material, 'color', attrib={'rgba': '{0} {1} {2} 1'.format(0.8, 0.8, 0.8)})

    visual_geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(visual_geometry, 'mesh', attrib={'filename': "../../all_meshes/cabinet.obj", 'scale': '{0} {1} {2}'.format(root_scale[0], root_scale[1], root_scale[2])})

    collision = ET.SubElement(base_link, 'collision')
    ET.SubElement(collision, 'origin', attrib={'xyz': '{0} {1} {2}'.format(root_position[0], root_position[1], root_position[2]), 'rpy': '{0} {1} {2}'.format(root_rot[0], root_rot[1], root_rot[2])})
    collision_geometry = ET.SubElement(collision, 'geometry')
    ET.SubElement(collision_geometry, 'mesh',  attrib={'filename': "../../all_meshes/cabinet.obj", 'scale': '{0} {1} {2}'.format(root_scale[0], root_scale[1], root_scale[2])})

    inertial = ET.SubElement(base_link, 'inertial')
    ET.SubElement(inertial, 'mass', attrib={'value': '1'})
    ET.SubElement(inertial, 'inertia', attrib={'ixx': '1e-4',  'ixy': '0', 'ixz': '0', 'iyy': '1e-4', 'iyz': '0', 'izz': '1e-4'})



    # Loop over links and create URDF elements for each
    link_names = ['base_link']
    for link_id, link_info in enumerate(links):
        link_names.append(os.path.basename(link_info)[:-4] + "{}".format(link_id))

    for link_id, link_info in enumerate(links):
        link_rot = Rot.from_quat(link_orientations[link_id]).as_rotvec()
        if "drawer" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 0.4
        elif "doorR" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 1.57
        elif "doorL" in os.path.basename(link_info)[:-4]:
            limitL = -1.57
            limitU = 0
        elif "doorD" in os.path.basename(link_info)[:-4]:
            limitL = -1.57
            limitU = 0
        elif "doorU" in os.path.basename(link_info)[:-4]:
            limitL = 0
            limitU = 1.57
        else:
            limitL = 0
            limitU = 1.57

        # Add a link element for each link
        # link_names.append(os.path.basename(link_info)[:-4]+"{}".format(link_id))
        link = ET.SubElement(robot, 'link', attrib={'name': os.path.basename(link_info)[:-4]+"{}".format(link_id)})
        visual = ET.SubElement(link, 'visual')
        ET.SubElement(visual, 'origin', attrib={'xyz': '0 0 0', 'rpy': '0 0 0'})
        material = ET.SubElement(visual, 'material', attrib={'name': 'white'})
        ET.SubElement(material, 'color', attrib={'rgba': '{0} {1} {2} 1'.format(0.8, 0.8, 0.8)})
        visual_geometry = ET.SubElement(visual, 'geometry')
        ET.SubElement(visual_geometry, 'mesh', attrib={'filename': "../../all_meshes/parts/{}".format(os.path.basename(link_info)), 'scale': '{0} {1} {2}'.format(link_scales[link_id][0], link_scales[link_id][1], link_scales[link_id][2])})

        collision = ET.SubElement(link, 'collision')
        ET.SubElement(collision, 'origin', attrib={'xyz': '0 0 0', 'rpy': '0 0 0'})
        collision_geometry = ET.SubElement(collision, 'geometry')
        ET.SubElement(collision_geometry, 'mesh', attrib={'filename': "../../all_meshes/parts/{}".format(os.path.basename(link_info)), 'scale': '{0} {1} {2}'.format(link_scales[link_id][0], link_scales[link_id][1], link_scales[link_id][2])})

        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', attrib={'value': '0.2'})
        ET.SubElement(inertial, 'inertia', attrib={'ixx': '1e-4', 'ixy': '0', 'ixz': '0', 'iyy': '1e-4', 'iyz': '0', 'izz': '1e-4'})

        # Add a joint element for each link
        joint = ET.SubElement(robot, 'joint', attrib={'name': '{0}_to_{1}'.format(link_names[link_id+1], link_names[linkparents[link_id]]), 'type': joint_names[jointtypes[link_id]]})
        ET.SubElement(joint, 'axis', attrib={'xyz': '{0} {1} {2}'.format(linkJointAxis[link_id][0], linkJointAxis[link_id][1], linkJointAxis[link_id][2])})
        ET.SubElement(joint, 'limit', attrib={'effort': '5', 'lower': '{0}'.format(limitL), 'upper':'{0}'.format(limitU), 'velocity':'2.283'})
        ET.SubElement(joint, 'origin', attrib={'xyz': '{0} {1} {2}'.format(link_positions[link_id][0], link_positions[link_id][1], link_positions[link_id][2]), 'rpy': '{0} {1} {2}'.format(link_rot[0], link_rot[1], link_rot[2])})

        ET.SubElement(joint, 'parent', attrib={'link': '{}'.format(link_names[linkparents[link_id]])})
        ET.SubElement(joint, 'child', attrib={'link': '{}'.format(link_names[link_id+1])})

    # Once all elements are added, write the URDF to a file
    tree = ET.ElementTree(robot)

    robot_xml_str = prettify(robot)
    with open('{}.urdf'.format(filename), 'w') as file:
        file.write(robot_xml_str)

    tree.write('{}.urdf'.format(filename), encoding='utf-8', xml_declaration=True)


def create_articulated_objects(root, root_scale, root_position, root_orientation, links, link_scales, link_positions,
                               link_orientations, linkparents, jointtypes, linkJointAxis, texture_list, random):
    if random:
        random_frame = True
        root, links, linkJointAxis, jointtypes = domain_randomization(root, links, random_frame, linkJointAxis,
                                                                      jointtypes)
    collisionIndices, visualIndices, link_positions = visual_collision_shapes(links, link_scales, link_positions)
    obj = p.createMultiBody(baseMass=0,
                            baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=root,
                                                                           meshScale=root_scale),
                            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_MESH, fileName=root,
                                                                     meshScale=root_scale),
                            basePosition=root_position,
                            baseOrientation=root_orientation,
                            linkMasses=[0.2] * len(jointtypes),
                            linkCollisionShapeIndices=collisionIndices,
                            linkVisualShapeIndices=visualIndices,
                            linkPositions=link_positions,
                            linkOrientations=link_orientations,
                            linkInertialFramePositions=[[0, 0, 0]] * len(jointtypes),
                            linkInertialFrameOrientations=[[0, 0, 0, 1]] * len(jointtypes),
                            linkParentIndices=linkparents,
                            linkJointTypes=jointtypes,
                            linkJointAxis=linkJointAxis
                            )

    if "meshes/oven_fan.obj" in root:
        color = [0.3, 0.3, 0.3, 1]
        p.changeVisualShape(obj, -1, rgbaColor=color)
    if len(texture_list) > 0:
        base_texture = os.path.dirname(texture_list[0]) + "/base.png"

        base_tex = p.loadTexture(base_texture)
        p.changeVisualShape(obj, -1, rgbaColor=(1, 1, 1, 1), textureUniqueId=base_tex)

    for i in range(p.getNumJoints(obj)):
        jointinfo = p.getJointInfo(obj, i)
        if len(texture_list) > 0:
            cab_texture = texture_list[int(jointinfo[12][4:]) - 1]
            cab_tex = p.loadTexture(cab_texture)
            p.changeVisualShape(obj, i, rgbaColor=(1, 1, 1, 1), textureUniqueId=cab_tex)

    return obj
