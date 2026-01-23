import os.path
import random
import shutil

import numpy as np

from utils import vis, utils
from functions import step_proc, img_proc
from functions.onshape import onshape_seq_parser
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import upper_funcs
from PIL import Image
from pathlib import Path
import requests
from requests.auth import HTTPBasicAuth
# from onshape_client.client import Client


## draw_para_net
# step_proc.step2pcd_batched_(r'C:\Users\ChengXi\Desktop\similar steps', 5000)
# step_proc.step2stl_batched_(r'C:\Users\ChengXi\Desktop\similar steps')
# vis.vis_mesh_view(r'C:\Users\ChengXi\Desktop\similar steps\eccentric wheel1.stl')
# vis.vis_pcd_view(r'C:\Users\ChengXi\Desktop\参数化与通用对比\chain_wheel3.txt', delimiter='\t', attr=-2)
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\图片处理')
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig')
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig\suppl')
# img_proc.remove_png_white_pixel(r'C:\Users\ChengXi\Desktop\fig\prisms for comp2.png')
# step_proc.step2pcd_faceseg(r'C:\Users\ChengXi\Desktop\gear-paper.STEP', 2500, r'C:\Users\ChengXi\Desktop\gear-paper.txt')
# vis.vis_pcd_view(r'C:\Users\ChengXi\Desktop\gear-paper.txt', -1)

## test
# file_name = ['D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000000\\00000000_290a9120f9f249a7a05cfe9c_step_000.step', 'D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000001\\00000001_1ffb81a71e5b402e966b9341_step_000.step', 'D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000002\\00000002_1ffb81a71e5b402e966b9341_step_001.step']
# for cstep in file_name:
#     vis.vis_step_file(cstep)
# vis.vis_step_file(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00000038\00000038_c7d977f326364e35bb5b5d27_step_001.step')
# img_proc.remove_png_white_pixel(r'E:\document\deeplearning_idea\参数化点云\参数化点云小论文\网站\cstnetwork.io\static\images\data_percentage.png')
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig', (255,255,255), 4)
# remove_u00a0(r'C:\Users\ChengXi\Desktop\60SJ.txt')
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\program')
# img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\valids\v2')
# 简单r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00000029\00000029_ad34a3f60c4a4caa99646600_step_009.step'
# vis.vis_data2d('log/loss-time.txt')
# adict = {'Geom_Plane': 1745610, 'Geom_CylindricalSurface': 439119, 'Geom_BSplineSurface': 294139, 'Geom_RectangularTrimmedSurface': 271824, 'Geom_ToroidalSurface': 64732, 'Geom_ConicalSurface': 43831, 'Geom_SurfaceOfLinearExtrusion': 43086, 'Geom_SphericalSurface': 23826, 'Geom_SurfaceOfRevolution': 20460, 'Geom_OffsetSurface': 2490}
# all_ins = 0
# for c_key in adict.keys():
#     all_ins += adict[c_key]
# print(f'instance_all: {all_ins}')
# plane_all = adict['Geom_Plane']
# cylinder_all = adict['Geom_CylindricalSurface']
# cone_all = adict['Geom_ConicalSurface'] * 4
# other_all = adict['Geom_BSplineSurface'] // 2
# all_ins = plane_all + cylinder_all + cone_all + other_all
# print(f'plane_count: {plane_all / all_ins}')
# print(f'cylinder_count: {cylinder_all / all_ins}')
# print(f'cone_count: {cone_all / all_ins}')
# print(f'other_count: {other_all / all_ins}')
# aval = 90.9104 + 99.8862 +77.5054 +79.5839 +155.2465 +89.0285
# bval = 45.2805+63.3458+49.0998+35.3037+45.4258+49.8011
# print(1 - bval / aval)

## main
# shape = step_proc.step_read_ocaf(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step')
# vis.vis_shapeocc(shape)
# aprtlist = step_proc.assemble_explode(r'C:\Users\ChengXi\Desktop\装配体1.STEP')
# aprtlist = step_proc.assemble_explode(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step')
#
# for c_shape in aprtlist:
#     vis.vis_shapeocc(c_shape)
# aroot = r'D:\document\DeepLearning\tmp\STEPMillion_pack1'
# files_all = utils.get_allfiles(aroot, 'step')
# assem = 0
# assem_all = []
# for c_file in tqdm(files_all, total=len(files_all)):
#     if step_proc.assembly_filter(c_file):
#         assem += 1
#         assem_all.append(c_file)
#         print(c_file)
# print(assem)
# with open('log/assems.txt', 'w') as f:
#     for cline in assem_all:
#         f.write(cline + '\n')
# print(step_proc.assembly_filter(r'C:\Users\ChengXi\Desktop\gear-paper.STEP'))
# print(step_proc.assembly_filter(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step'))


def is_all_step_transed(source_dir=r'F:\document\deeplearning\Param20K_Extend', target_dir=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend'):
    """
    判断是否全部的step文件都进行了转换
    :return:
    """
    all_step = utils.get_allfiles(source_dir, 'STEP')

    for c_step in tqdm(all_step):
        pcd_path = c_step.replace(source_dir, target_dir)
        pcd_path = os.path.splitext(pcd_path)[0] + '.txt'

        if not os.path.isfile(pcd_path):
            print(c_step)


def is_npnt_sufficient():
    """
    判断点数是否满足要求
    :return:
    """
    target_dir = r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend'
    fils_all = utils.get_allfiles(target_dir)
    for c_file in tqdm(fils_all):
        c_pnd = np.loadtxt(c_file)
        c_num = c_pnd.shape[0]

        if c_num < 2000:
            print(c_num, c_file)


def source_to(source_dir=r'F:\document\deeplearning\Param20K_Extend', target_dir=r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend', log_file=r'C:\Users\ChengXi\Desktop\cstnet2\not_suff.txt'):
    """
    补救一些数量不够的数据
    :param source_dir:
    :param target_dir:
    :param log_file:
    :return:
    """
    with open(log_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]

    for c_idx, c_pcd in enumerate(lines):

        over_rate = 0.025
        while True:

            print(f'{c_idx} / {len(lines)}')

            c_step = c_pcd.replace(target_dir, source_dir)
            c_step = os.path.splitext(c_step)[0] + '.STEP'

            real_saved = step_proc.step2pcd(c_step, c_pcd, 2000, print_log=False, over_rate=over_rate)

            if real_saved >= 2000:
                break
            else:
                over_rate += 0.1
                print(f'point saved: {real_saved}, resample over rate: {over_rate}')


def update_loc(pmt, loc, mad, trans):
    """
    更新几何体loc

    参数:
        pmt: [n, ] numpy数组，几何体类型 (0=plane, 1=cylinder, 其它=直接平移)
        loc: [n, 3] numpy数组，原点到几何体的垂足
        mad: [n, 3] numpy数组，几何体的法线/轴向（需为单位向量）
        translation: (a, b, c) 平移向量
    返回:
        loc_updated: [n, 3] numpy数组，更新后的垂足
    """
    loc_updated = loc.copy()

    # ---------- 平面 ----------
    mask_plane = (pmt == 0)
    if np.any(mask_plane):
        n = mad[mask_plane]
        proj = np.sum(trans * n, axis=1, keepdims=True) * n
        loc_updated[mask_plane] = loc[mask_plane] + proj

    # ---------- 圆柱 ----------
    mask_cyl = (pmt == 1)
    if np.any(mask_cyl):
        d = mad[mask_cyl]
        proj = np.sum(trans * d, axis=1, keepdims=True) * d
        loc_updated[mask_cyl] = loc[mask_cyl] + (trans - proj)

    # ---------- 其它 ----------
    mask_other = ~(mask_plane | mask_cyl)
    if np.any(mask_other):
        loc_updated[mask_other] = loc[mask_other] + trans

    return loc_updated


def update_dim(pmt, dim, scale):
    # 找到圆锥
    mask_plane = (pmt == 2)
    other_pmt = ~mask_plane
    dim[other_pmt] = dim[other_pmt] * scale

    return dim


def single_load(pcd_file):
    point_set = np.loadtxt(pcd_file)

    xyz = point_set[:, :3]  # [n, 3]
    pmt = point_set[:, 3].astype(np.int32)  # 基元类型 [n, ]
    mad = point_set[:, 4:7]  # 主方向 [n, 3]
    dim = point_set[:, 7]  # 主尺寸 [n, ]
    nor = point_set[:, 8:11]  # 法线 [n, 3]
    loc = point_set[:, 11:14]  # 主位置 [n, 3]
    affil_idx = point_set[:, 14]  # 从属索引 [n, ]

    # 质心平移到原点，三轴范围缩放到 [-1, 1]^3
    move_dir = -np.mean(xyz, axis=0)
    xyz = xyz + move_dir
    scale = 1.0 / np.max(np.sqrt(np.sum(xyz ** 2, axis=1)), 0)
    xyz = xyz * scale

    # 平移缩放后，pmt, mad, nor 不变，dim 除圆锥外与原本进行相同比例缩放，loc 先平移，再缩放
    dim = update_dim(pmt, dim, scale)
    loc = update_loc(pmt, loc, mad, move_dir)
    loc = loc * scale

    return xyz, pmt, mad, dim, nor, loc, affil_idx


def process_cannot_convert():
    log_file = r'C:\Users\ChengXi\Desktop\cstnet2\not_trans.txt'
    source_dir = r'F:\document\deeplearning\Param20K_Extend'
    target_dir = r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend'

    fail_files = []

    with open(log_file, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file]

    for c_line in lines:
        if 'cannot convert this STEP file:' in c_line:
            c_fail_step = c_line.split('STEP file: ')[1][:-1]
            fail_files.append(c_fail_step)
            print(c_fail_step)

    for idx, c_step in enumerate(fail_files):
        # if idx <= 18:
        #     continue

        pcd_path = c_step.replace(source_dir, target_dir)
        pcd_path = os.path.splitext(pcd_path)[0] + '.txt'

        print(f'[{idx} / {len(fail_files)}] 当前处理：{c_step}')
        step_proc.step2pcd(c_step, pcd_path, 2000, 1e-4)


def del_except(target_dir=r'F:\document\deeplearning\Param20K_Extend', suffix='.STEP'):
    files_all = utils.get_allfiles(target_dir, None)

    for c_file in tqdm(files_all):
        c_suffix = os.path.splitext(c_file)[1]
        if c_suffix != suffix:
            os.remove(c_file)


def divided_to_sketch_and_photo():
    source_dir = r'D:\document\DeepLearning\DataSet\草图项目\sketch_and_shortcut'
    target_dir = r'D:\document\DeepLearning\DataSet\草图项目\retrieval_cad'

    # 获取全部类别


def vis_pcd_gen():
    # 生成测试的点云
    # stepfile = r'C:\Users\ChengXi\Desktop\cstnet2\cube_div.STEP'
    # stepfile = r'F:\document\deeplearning\Param20K_Extend\test\bearing\01447962.STEP'
    stepfile = r'D:\document\DeepLearning\DataSet\STEP_All\Param20K_STEP\train\rivet\trans13120.STEP'
    pcd_file = r'C:\Users\ChengXi\Desktop\cstnet2\comb.txt'
    #
    step_proc.step2pcd(stepfile, pcd_file, 2000, is_normalize=False)
    #
    # pcd_file = r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend\test\bearing\00042353.txt'
    pnts_all = np.loadtxt(pcd_file)
    #
    xyz = pnts_all[:, :3]
    pmt = pnts_all[:, 3].astype(np.int32)
    mad = pnts_all[:, 4:7]
    dim = pnts_all[:, 7]
    nor = pnts_all[:, 8:11]
    loc = pnts_all[:, 11:14]
    affil_idx = pnts_all[:, 14].astype(np.int32)

    cone_mask = (pmt == 2)
    if cone_mask.sum() != 0:
        cone_mad = mad[cone_mask]
        cone_apex = loc[cone_mask]

        t = - np.einsum('ij,ij->i', cone_mad, cone_apex)

        # 垂足坐标
        perpendicular_foot = cone_apex + t[:, None] * cone_mad
        loc[cone_mask] = perpendicular_foot

    print(f'xmax {xyz[:, 0].max()}, xmin {xyz[:, 0].min()}, ymax {xyz[:, 1].max()}, ymin {xyz[:, 1].min()}, zmax {xyz[:, 2].max()}, zmin {xyz[:, 2].min()}')

    show_pnt_attr(xyz, affil_idx)
    # vis.vis_pcd_with_attr(xyz, None, affil_idx)
    # vis.vis_pcd_plt(xyz, loc)


def convert_jpg_to_png(root_folder, overwrite=False):
    """
    将 root_folder 及子文件夹中所有 JPG 转为 PNG

    Parameters:
        root_folder (str): 根目录
        overwrite (bool): 是否覆盖已存在的 PNG 文件
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg")):
                jpg_path = os.path.join(dirpath, filename)
                png_path = os.path.join(dirpath, filename.rsplit(".", 1)[0] + ".png")

                # 如果 PNG 已存在且不覆盖，则跳过
                if not overwrite and os.path.exists(png_path):
                    print(f"跳过（已存在）：{png_path}")
                    continue

                try:
                    img = Image.open(jpg_path)
                    img.save(png_path)
                    print(f"转换成功：{jpg_path} -> {png_path}")
                except Exception as e:
                    print(f"转换失败：{jpg_path}, 错误：{e}")


def acc_correct():
    n_chair = 323
    n_shoe = 666

    chair_vals = [0.220,0.368,0.235,0.377,0.238,0.385,0.241,0.407,0.255,0.437,0.252,0.440,0.725,0.807,]
    shoe_vals = [0.154,0.300,0.158,0.307,0.164,0.316,0.176,0.352,0.179,0.355,0.192,0.389,0.455,0.715,]

    chair_vals_new = [int(x * n_chair) / n_chair for x in chair_vals]
    shoe_vals_new = [int(x * n_shoe) / n_shoe for x in shoe_vals]

    print(chair_vals_new)
    print(shoe_vals_new, '\n')

    for i in range(len(chair_vals_new) // 2):
        # print(f'{chair_vals_new[2*i] }, {chair_vals_new[2*i+1]}, {shoe_vals_new[2*i]}, {shoe_vals_new[2*i+1]}')
        print(f'{chair_vals_new[2 * i] * n_chair}, {chair_vals_new[2 * i + 1] * n_chair}, {shoe_vals_new[2 * i] * n_shoe}, {shoe_vals_new[2 * i + 1] * n_shoe}')


def acc_correct2():
    base_num = 34500
    # all_val = [0.6808,0.6908,0.7422,0.6924,0.7050,0.7310,0.6068,0.6665,0.6224,0.6768,0.6800,0.6977,0.5249,0.7070,0.7280,0.6829,0.7537]
    # all_val = [0.6768,0.6881,0.6759,0.6665,0.6714,0.6631,0.7280,0.7273,0.7291,0.6829,0.6802,0.6795]
    all_val = [0.7249, 0.6881, 0.6759, 0.6665, 0.6714, 0.6631, 0.7280, 0.7273, 0.7291, 0.6829, 0.6802, 0.6795]

    # base_num = 12694
    # all_val = [0.490,0.349,0.623,0.658,0.279,0.339,0.293,0.344,0.403,0.441,0.669]

    all_val_new = [int(x * base_num) / base_num for x in all_val]

    for c_val, c_val_new in zip(all_val, all_val_new):
        print(f'{c_val}, {c_val_new: .5f}')


def sketch_proj_select_subset(src_folder, dst_folder, max_nsave_cat = 10):
    """
    用于草图项目，在数据集中选择一个子集
    :param src_folder:
    :param dst_folder:
    :param max_nsave_cat:
    :return:
    """

    # 在目标文件夹下创建类似的文件夹结构
    utils.create_tree_like(src_folder, dst_folder)

    # 获取全部类别
    cls_all = utils.get_subdirs(os.path.join(src_folder, 'model_3d'))

    for c_cls in cls_all:
        c_save_cat = 0

        c_step_dir = os.path.join(src_folder, 'model_3d', c_cls)
        c_steps_all = utils.get_allfiles(c_step_dir, 'step')

        c_ids = []
        for cc_step in c_steps_all:
            c_base_name = utils.basename_without_ext(cc_step)
            c_ids.append(c_base_name)

        # 找到对应的图片和草图的路径
        c_img_dir = os.path.join(src_folder, 'photo', c_cls)
        c_skh_dir = os.path.join(src_folder, 'sketch_s3_352', c_cls)

        c_imgs_all = utils.get_allfiles(c_img_dir, 'png')
        c_skhs_all = utils.get_allfiles(c_skh_dir, 'txt')

        for c_id in c_ids:
            c_img_path = os.path.join(c_img_dir, c_id + '_1.png')
            c_skh_path = os.path.join(c_skh_dir, c_id + '_1.txt')
            c_stp_path = os.path.join(c_step_dir, c_id + '.STEP')

            if c_img_path in c_imgs_all and c_skh_path in c_skhs_all:
                # 复制文件
                c_img_target = c_img_path.replace(src_folder, dst_folder)
                c_skh_target = c_skh_path.replace(src_folder, dst_folder)
                c_stp_target = c_stp_path.replace(src_folder, dst_folder)

                shutil.copy(c_img_path, c_img_target)
                shutil.copy(c_skh_path, c_skh_target)
                shutil.copy(c_stp_path, c_stp_target)

                c_save_cat += 1
                if c_save_cat >= max_nsave_cat:
                    break


def test_parse_onshape():
    seq_file = './shapes/cover.yml'
    save_file = r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\cover.STEP'

    onshape_seq_parser.seq_to_step(seq_file, save_file)


def test_save_onshape():
    onshape_seq_parser.save_part()


def get_default_plane_id():
    # did = "0dde8d689fcce59931c8fbca"
    # eid = "5ae1071dadf8820769e721e3"
    # micro = "7d3bcebac30a85a35a33b1fe"
    #
    # url = f"https://cad.onshape.com/api/partstudios/d/{did}/m/{micro}/e/{eid}/topology"
    #
    #
    # resp = requests.get(url, auth=HTTPBasicAuth(onshape_seq_parser.ACCESS_KEY, onshape_seq_parser.SECRET_KEY))
    # data = resp.json()
    #
    # for face in data.get("faces", []):
    #     print(face["id"], face.get("surface", {}).get("type"), face.get("featureId"), face.get("name"))

    client = Client(configuration={
        "base_url": "https://cad.onshape.com",
        "access_key": onshape_seq_parser.ACCESS_KEY,
        "secret_key": onshape_seq_parser.SECRET_KEY
    })

    print('clint id:', client.get_authorization_url())

    did = "36fc37575b42a9da77cce228"
    wid = "aca639292b8175da0cee3ac3"
    eid = "50251856ed10b39c9dc4e4a9"  # Part Studio
    L = 50  # 正方体边长（mm）

    # -----------------------------
    # 1. 构造草图特征（Sketch）
    # -----------------------------
    sketch_feature = {
        "feature": {
            "featureType": "sketch",
            "name": "SquareSketch",
            "parameters": [
                {
                    "message": {
                        "parameterId": "sketchPlane",
                        "nodeId": "sketchPlane",
                        "value": {
                            "message": {
                                "entityType": "BTMEntity",
                                "id": "JDC",  # Top plane（默认基准面）
                                "type": "FACE"
                            }
                        }
                    }
                }
            ],
            "entities": [
                # ---- 4 条边（正方形）----
                {
                    "entityType": "BTMSketchCurveSegment",
                    "entityId": "L1",
                    "geometry": {
                        "type": "line",
                        "startPoint": [0, 0],
                        "endPoint": [L, 0]
                    }
                },
                {
                    "entityType": "BTMSketchCurveSegment",
                    "entityId": "L2",
                    "geometry": {
                        "type": "line",
                        "startPoint": [L, 0],
                        "endPoint": [L, L]
                    }
                },
                {
                    "entityType": "BTMSketchCurveSegment",
                    "entityId": "L3",
                    "geometry": {
                        "type": "line",
                        "startPoint": [L, L],
                        "endPoint": [0, L]
                    }
                },
                {
                    "entityType": "BTMSketchCurveSegment",
                    "entityId": "L4",
                    "geometry": {
                        "type": "line",
                        "startPoint": [0, L],
                        "endPoint": [0, 0]
                    }
                }
            ],
            "constraints": [
                # ---- 端点重合，形成闭环 ----
                {"constraintType": "COINCIDENT", "entities": ["L1.end", "L2.start"]},
                {"constraintType": "COINCIDENT", "entities": ["L2.end", "L3.start"]},
                {"constraintType": "COINCIDENT", "entities": ["L3.end", "L4.start"]},
                {"constraintType": "COINCIDENT", "entities": ["L4.end", "L1.start"]},

                # ---- 平行约束 ----
                {"constraintType": "PARALLEL", "entities": ["L1", "L3"]},
                {"constraintType": "PARALLEL", "entities": ["L2", "L4"]}
            ],
            "dimensions": [
                # ---- 尺寸：L1 长度 = L ----
                {
                    "entityType": "BTMSketchDimension",
                    "entityId": "D1",
                    "dimensionType": "DISTANCE",
                    "entities": ["L1.start", "L1.end"],
                    "value": L
                }
            ]
        }
    }

    # client.part_studios_api.add_part_studio_feature
    client.part_studios_api.add_part_studio_feature(
        did=did, wvm='w', wvmid=wid, eid=eid,
        body=sketch_feature
    )

    print("Sketch with constraints and dimension created.")

    # -----------------------------
    # 2. 构造拉伸特征（Extrude）
    # -----------------------------
    extrude_feature = {
        "feature": {
            "featureType": "extrude",
            "name": "CubeExtrude",
            "parameters": [
                {
                    "message": {
                        "parameterId": "entities",
                        "value": {
                            "message": {
                                "entityType": "BTMEntity",
                                "id": "SquareSketch",
                                "type": "FACE"
                            }
                        }
                    }
                },
                {
                    "message": {
                        "parameterId": "depth",
                        "value": L
                    }
                },
                {
                    "message": {
                        "parameterId": "operationType",
                        "value": "NEW"
                    }
                }
            ]
        }
    }

    client.part_studios_api.add_feature(
        did=did, wvm='w', wvmid=wid, eid=eid,
        body=extrude_feature
    )

    print("Extrude completed. Cube created.")


    # client = Client(
    #     configuration={
    #         'base_url': 'https://cad.onshape.com',
    #         'access_key': onshape_seq_parser.ACCESS_KEY,
    #         'secret_key': onshape_seq_parser.SECRET_KEY
    #     }
    # )
    #
    # did = 'DOCUMENT_ID'
    # wid = 'WORKSPACE_ID'
    #
    # elements = client.documents_api.get_elements(did=did, wvm='w', wvmid=wid)
    # for e in elements:
    #     print(e['name'], e['id'], e['elementType'])


def test_onshape_parser():
    # onshape_seq_parser.parse_onshape_topology()
    onshape_seq_parser.test_parse_sketch()
    # from functions.onshape import onshape_seq_parser_old as od
    # od.test()


def vis_cst_pcd():
    """
    可视化某个文件夹下的带约束的点云
    :return:
    """
    target_dir = r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend\train'

    # 获取全部文件
    all_pcd = utils.get_allfiles(target_dir)

    # 打乱文件
    random.shuffle(all_pcd)

    for c_pcd in all_pcd:
        vis_cst_pcd_single(c_pcd)


def vis_cst_pcd_single(target_pcd):
    """
    可视化某个文件夹下的带约束的点云
    :return:
    """
    pnts_all = np.loadtxt(target_pcd)

    xyz = pnts_all[:, :3]
    pmt = pnts_all[:, 3].astype(np.int32)
    mad = pnts_all[:, 4:7]
    dim = pnts_all[:, 7]
    nor = pnts_all[:, 8:11]
    loc = pnts_all[:, 11:14]
    affil_idx = pnts_all[:, 14].astype(np.int32)

    # vis.vis_pcd_with_attr(xyz, None, affil_idx)
    show_pnt_attr(xyz, affil_idx, target_pcd)


def show_pnt_attr(points, attr, title=None):
    # n = 1000
    # points = np.random.randn(n, 3)
    # attr = np.random.randint(0, 10, size=n)  # 离散属性

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=attr,              # 关键：属性作为颜色
        cmap='tab10',        # 离散属性推荐 tab10 / tab20
        s=5                  # 点大小
    )

    plt.colorbar(sc, ax=ax, label='Attribute')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    set_axes_equal(ax)

    plt.show()


def set_axes_equal(ax):
    """
    Make 3D axes have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == '__main__':
    # img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig', (255, 255, 255), 4)
    # vis.vis_cls_log(r'C:\Users\ChengXi\Desktop\cstnet2\pnet2_geomloss.txt')
    # vis.vis_step_cloud(r'C:\Users\ChengXi\Desktop\apart.STEP', color=[38,40,46], n_points=700)
    # step_proc.test()

    # is_npnt_sufficient()
    # is_all_step_transed()
    # source_to()

    # ashape_occ = step_proc.step_read_ocaf(stepfile)
    # ashape_occ = step_proc.normalize_shape_to_unit_cube(ashape_occ)
    # step_proc.shapeocc2step(ashape_occ, r'C:\Users\ChengXi\Desktop\cstnet2\comb---2.STEP')

    # step_proc.step2pcd_batched(r'D:\document\DeepLearning\DataSet\STEP_All\Param20K_STEP', r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend2', 2000, 1e-4, 8)
    # process_cannot_convert()
    # print(os.path.splitext(stepfile))
    # del_except()

    # upper_funcs.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig\tvcg_rev', remove_pixel=(240,240,240))
    # upper_funcs.remove_png_white_pixel_batched(
    #     r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal_fig_save\generation\sketch-lattice',
    #     remove_pixel=(255, 255, 255))
    # vis_pcd_gen()
    # vis.vis_pcd(r'D:\document\DeepLearning\DataSet\MCB_PointCloud\MCB_A\train\Helical geared motors\00025702.txt', delimiter=' ')

    # convert_jpg_to_png(r'E:\document\deeplearning_idea\sketch temporal is out\fid_cal_fig_save\generation', True)

    # acc_correct()
    # utils.unify_step_suffix_recursive(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketch_cad\model_3d')

    # sketch_proj_select_subset(r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketch_cad', r'D:\document\DeepLearning\DataSet\sketch_retrieval\sketch_cad_small')

    # acc_correct2()

    # test_parse_onshape()

    # test_deepcad_onshape_parser()4R2437.txt

    # vis_cst_pcd()
    # vis_cst_pcd_single(r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend\train\rivet\trans13120.txt')
    # vis_pcd_gen()

    test_onshape_parser()


    pass






