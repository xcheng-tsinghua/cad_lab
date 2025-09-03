# pack
import os
import shutil
import numpy as np
from pathlib import Path
import open3d as o3d
import pymeshlab
import py7zr
# import step_proc


def is_suffix_step(filename):
    if filename[-4:] == '.stp' \
            or filename[-5:] == '.step' \
            or filename[-5:] == '.STEP':
        return True

    else:
        return False


def rename_files_in_dir(dir_path, start_ind=0):

    def rename_raw(start_local):
        # 获取文件夹中所有文件的列表
        files = os.listdir(dir_path)

        # 对文件进行排序，以确保按顺序重命名
        # files.sort()

        # 初始化新的文件名计数器
        counter = start_local

        # 遍历文件夹中的每个文件
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(dir_path, file)

            # 检查是否为文件（排除子文件夹）
            if os.path.isfile(file_path):
                # 构造新的文件名
                new_file_name = f'{counter}.txt'
                # 构造新的文件完整路径
                new_file_path = os.path.join(dir_path, new_file_name)

                # 重命名文件
                os.rename(file_path, new_file_path)
                # print(f"Renamed {file} to {new_file_name}")

                # 增加计数器
                counter += 1

    print('rename files')
    rename_raw(9999999)
    rename_raw(start_ind)


def get_subdirs(dir_path):
    """
    获取 dir_path 的所有一级子文件夹
    仅仅是文件夹名，不是完整路径
    """
    path_allclasses = Path(dir_path)
    directories = [str(x) for x in path_allclasses.iterdir() if x.is_dir()]
    dir_names = [item.split(os.sep)[-1] for item in directories]

    return dir_names


def create_subdirs(dir_path, sub_names: list):
    """
    在 dir_path 下创建多个一级子文件夹
    dir_path
     ├─ sub_names[0]
     ├─ sub_names[1]
     ├─ sub_names[2]
     ├─ ...
     └─ sub_names[-1]

    :param dir_path: 目标文件夹
    :param sub_names: 全部子文件夹名
    :return: void
    """
    for dir_name in sub_names:
        os.makedirs(os.path.join(dir_path, dir_name), exist_ok=True)


def create_tree_like(source_dir, target_dir):
    """
    在target_dir下创建与source_dir相同的目录层级
    :param source_dir:
    :param target_dir:
    :return:
    """
    for root, dirs, files in os.walk(source_dir):
        # 计算目标文件夹中的对应路径
        relative_path = os.path.relpath(root, source_dir)
        target_path = os.path.join(target_dir, relative_path)

        # 创建目标文件夹中的对应目录
        os.makedirs(target_path, exist_ok=True)


def get_allfiles(dir_path, suffix='txt', filename_only=False):
    '''
    获取dir_path下的全部文件路径
    '''
    filepath_all = []

    def other_judge(file_name):
        if file_name.split('.')[-1] == suffix:
            return True
        else:
            return False

    if suffix == 'stp' or suffix == 'step' or suffix == 'STEP':
        suffix_judge = is_suffix_step
    else:
        suffix_judge = other_judge

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if suffix_judge(file):
                if filename_only:
                    current_filepath = file
                else:
                    current_filepath = str(os.path.join(root, file))
                filepath_all.append(current_filepath)

    return filepath_all


def del_file_by_suffix(dir_path, suffix='txt'):
    """
    删除文件夹内指定后缀的文件
    """
    # 先找到目标文件夹下全部文件
    files_all = get_allfiles(dir_path)

    delete_count = 0
    for file_pth in files_all:

        if file_pth.split('.')[-1] == suffix:
            try:
                os.remove(file_pth)
                print(f'文件 {file_pth} 已成功删除。')
                delete_count += 1
            except FileNotFoundError:
                print(f'文件 {file_pth} 不存在。')
            except PermissionError:
                print(f'没有权限删除文件 {file_pth}。')
            except Exception as e:
                print(f'删除文件 {file_pth} 时发生错误: {e}')

    print('删除文件数：', delete_count)


def remove_path_suffix(file_path):
    """
    删除文件的路径和后缀名
    :param file_path:
    :return:
    """
    return os.path.splitext(os.path.basename(file_path))[0]


