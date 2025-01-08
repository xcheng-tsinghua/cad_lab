# open cascade
from OCC.Display.SimpleGui import init_display
# others
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
# self
import step_proc


def vis_pcd(filepath, attr_show=None, show_normal=False):
    data_all = np.loadtxt(filepath)

    pcd = o3d.geometry.PointCloud()
    points = data_all[:, 0:3]
    pcd.points = o3d.utility.Vector3dVector(points)

    if show_normal:
        normals = data_all[:, 3: 6]
        pcd.normals = o3d.utility.Vector3dVector(normals)

    if attr_show is not None:
        labels = data_all[:, attr_show]
        unique_labels = np.unique(labels)
        num_labels = len(unique_labels)
        colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], point_show_normal=show_normal)


def vis_step_cloud(file_name, n_points=2500, with_cst=False, attr_show=None, deflection=0.1, tmp_stl='tmp/tmp.stl', tmp_pc='tmp/tmp_pc.txt'):
    '''
    将STEP文件转化为点云之后进行可视化
    '''
    step_proc.step2pcd(file_name, n_points, tmp_pc, deflection, not with_cst)
    vis_pcd(tmp_pc, attr_show)

    os.remove(tmp_pc)


def vis_shapeocc(shape_occ):
    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape_occ, update=True)

    start_display()


def vis_step_file(file_name):
    vis_shapeocc(step_proc.step_read_ocaf(file_name))


def get_o3d_viewangle_json(json_file='tmp/camera_params.json'):
    with open(json_file, 'r') as f:
        camera_params = json.load(f)

    # 获取外参矩阵（extrinsic matrix）
    extrinsic_matrix = np.array(camera_params['extrinsic']).reshape([4, 4])

    # 从外参矩阵中提取旋转矩阵
    rotation_matrix = extrinsic_matrix[:3, :3]

    # 前方向（相机朝向）
    front = -rotation_matrix[:, 2]  # 取第三列并取反
    front = front / np.linalg.norm(front)  # 归一化

    # 上方向
    up = rotation_matrix[:, 1]  # 取第二列
    up = up / np.linalg.norm(up)  # 归一化

    print('front: ', front)
    print('up: ', -up)

    return front, -up


def vis_pcd_view(pcd_path, attr=None, show_normal=False, delimiter='\t'):

    data_all = np.loadtxt(pcd_path, delimiter=delimiter)
    pcd = o3d.geometry.PointCloud()
    points = data_all[:, 0:3]
    pcd.points = o3d.utility.Vector3dVector(points)

    if show_normal:
        normals = data_all[:, 3: 6]
        normals = (normals + 1) / 2
        pcd.colors = o3d.utility.Vector3dVector(normals)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones_like(normals))
        # pcd.normals = o3d.utility.Vector3dVector(normals)

    if attr is not None:
        labels = data_all[:, attr]

        if attr == -1:  # 基元类型
            # num_labels = 4
            # colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap

            labels = data_all[:, attr]
            unique_labels = np.unique(labels)
            num_labels = len(unique_labels)
            colors = np.array([plt.cm.tab10(label / num_labels) for label in labels])[:, :3]  # Using tab10 colormap
            pcd.colors = o3d.utility.Vector3dVector(colors)

        elif attr == -2:  # 是否邻近边
            colors = []
            for c_attr in labels:
                if c_attr == 0:
                    # colors.append((0, 0, 0))
                    # colors.append((255, 215, 0))
                    colors.append((189 / 255, 216 / 255, 232 / 255))
                    # colors.append((60 / 255, 84 / 255, 135 / 255))

                elif c_attr == 1:
                    # colors.append((255, 215, 0))
                    # colors.append((0, 0, 0))
                    colors.append((19 / 255, 75 / 255, 108 / 255))
                    # colors.append((230 / 255, 75 / 255, 52 / 255))

                else:
                    raise ValueError('not valid edge nearby')

            colors = np.array(colors)

        else:
            raise ValueError('not valid attr')

        pcd.colors = o3d.utility.Vector3dVector(colors)

    else:
        colors = []

        for _ in points[:, 0]:
            colors.append((189 / 255, 216 / 255, 232 / 255))

        pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    view_control = vis.get_view_control()
    v_front, v_up = get_o3d_viewangle_json()
    view_control.set_front(v_front)
    view_control.set_up(v_up)

    view_control.set_lookat([0, 0, 0])
    view_control.set_zoom(0.7)

    vis.update_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_show_normal = show_normal
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    # 保存相机参数到json文件
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("tmp/camera_params.json", camera_params)

    vis.destroy_window()


def vis_mesh_view(mesh_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0., 1., 1.])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    origin_pos = [0, 0, 0]
    view_control = vis.get_view_control()

    v_front, v_up = get_o3d_viewangle_json()
    view_control.set_front(v_front)
    view_control.set_up(v_up)
    view_control.set_lookat(origin_pos)
    view_control.set_zoom(3)

    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    # 保存相机参数到json文件
    # camera_params = view_control.convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters("camera_params.json", camera_params)

    vis.destroy_window()


def vis_data2d(file_name, delimiter=','):
    """
    该文本文件的每行需要是 x,y
    :param file_name:
    :param delimiter:
    :return:
    """
    points = np.loadtxt(file_name, delimiter=delimiter)

    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], s=2)
    plt.show()






