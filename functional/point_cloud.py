import os
from tqdm import tqdm
import re

from functional import utils


def del_pcd_by_pointcount(dir_path, cmd='<2000'):

    # 先找到目标文件夹下全部文件
    files_all = utils.get_allfiles(dir_path)

    delete_count = 0
    for file_pth in tqdm(files_all, total=len(files_all)):

        with open(file_pth, 'r') as f:
            lines = f.readlines()
            line_count = len(lines)

        if eval(str(line_count) + cmd):
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


def is_cst_format_fit(dir_path):
    """
    判断 dir_path 内的全部txt文件每行是否符合格式: float \t float \t float \t float \t float \t float \t int \t int
    """

    def check_format(line):
        # 正则表达式检查行的格式
        pattern = re.compile(r'^([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'([-+]?\d*\.?\d+(e[-+]?\d+)?|\d+)\t'
                             r'(\d+)\t'
                             r'(\d+)$')
        match = pattern.match(line.strip())
        return bool(match)

    print('检查文件格式')
    format_fit = True
    not_fit_count = 0

    files_all = utils.get_allfiles(dir_path)

    for c_file in tqdm(files_all, total=len(files_all)):
        with open(c_file, 'r') as f:
            for c_line in f.readlines():
                if not check_format(c_line):
                    print(c_file, '不符合点云文件格式')
                    print(c_line.strip())
                    format_fit = False
                    not_fit_count += 1
                    break

    print('not fit all:', not_fit_count)
    return format_fit






