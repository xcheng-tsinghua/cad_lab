import os
import open3d as o3d
import pymeshlab
import numpy as np
import shutil

from utils import get_allfiles, get_subdirs


def get_points_o3d(mesh_file, n_points, save_path):
    save_path = os.path.abspath(save_path)

    try:
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        if mesh.has_triangles() is True:
            target = mesh.sample_points_uniformly(number_of_points=n_points, use_triangle_normal=True)
        else:
            raise ValueError
    except:
        print('Failed to read Mesh')
        try:
            target = o3d.io.read_point_cloud(mesh_file)
        except:
            print('Failed to read Point Cloud Data')
            target = False

    points = target.points
    normals = target.normals
    with open(save_path, 'w') as f:
        for i in range(10000):
            point = points[i]
            normal = normals[i]
            line = '{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n'.format(point[0], point[1], point[2], normal[0], normal[1], normal[2])
            f.write(line)

    return target


def get_points_mslab(mesh_file, n_points, save_path=None):

    # 加载OBJ文件
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_file)

    # 生成点云
    ms.generate_sampling_poisson_disk(samplenum=n_points)

    # 获取点云数据和法向量
    vertex_matrix = ms.current_mesh().vertex_matrix()
    normal_matrix = ms.current_mesh().vertex_normal_matrix()
    data = np.hstack((vertex_matrix, normal_matrix))

    if save_path is not None:
        save_path = os.path.abspath(save_path)
        # 保存点云数据和法向量
        np.savetxt(save_path, data, fmt='%.6f', delimiter=' ')

    return data


def batched_mesh_to_pcd(source_dir=r'D:\document\DeepLearning\DataSet\MCB\MCB_A\train', target_dir=r'D:\document\DeepLearning\paper_draw\AttrVis_MCB2', k=5):
    classes = get_subdirs(source_dir)

    for idx, c_class in enumerate(classes):
        print(c_class, f'{idx}/{len(classes)}')
        c_class_dir = os.path.join(source_dir, c_class)
        all_meshes = get_allfiles(c_class_dir, 'obj')

        for i in range(k):
            c_mesh = all_meshes[i]

            base_mesh = os.path.basename(c_mesh)

            tar_mesh = os.path.join(target_dir, base_mesh)
            tar_pcd = tar_mesh.replace('.obj', '.txt')

            shutil.copy(c_mesh, tar_mesh)
            get_points_mslab(tar_mesh, 3000, tar_pcd)


if __name__ == '__main__':
    batched_mesh_to_pcd()
    pass

















