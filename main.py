import os.path

import numpy as np

from utils import vis, utils
from processor import step_proc, img_proc
from tqdm import tqdm





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

def filt():
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


if __name__ == '__main__':
    # img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig', (255, 255, 255), 4)
    # vis.vis_cls_log(r'C:\Users\ChengXi\Desktop\cstnet2\pnet2_geomloss.txt')
    # vis.vis_step_cloud(r'C:\Users\ChengXi\Desktop\apart.STEP', color=[38,40,46], n_points=700)
    # step_proc.test()

    # 生成测试的点云
    stepfile = r'C:\Users\ChengXi\Desktop\cstnet2\comb.STEP'
    pcd_file = r'C:\Users\ChengXi\Desktop\cstnet2\comb.txt'
    step_proc.step2pcd(stepfile, pcd_file, 2000)

    pnts_all = np.loadtxt(pcd_file)

    xyz = pnts_all[:, :3]
    pmt = pnts_all[:, 3]
    mad = pnts_all[:, 4:7]
    dim = pnts_all[:, 7]
    nor = pnts_all[:, 8:11]
    loc = pnts_all[:, 11:14]
    affil_idx = pnts_all[:, 14]
    #
    # xyz = np.vstack([xyz, loc])
    # vis.vis_pcd_with_attr(xyz, nor, pmt)

    # step_proc.step2pcd_batched(r'F:\document\deeplearning\Param20K_Extend', r'D:\document\DeepLearning\DataSet\pcd_cstnet2\Param20K_Extend', workers=8)

    # filt()
    # source_to()

    # ashape_occ = step_proc.step_read_ocaf(stepfile)
    # ashape_occ = step_proc.normalize_shape_to_unit_cube(ashape_occ)
    # step_proc.shapeocc2step(ashape_occ, r'C:\Users\ChengXi\Desktop\cstnet2\comb---2.STEP')





    pass






