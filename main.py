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


if __name__ == '__main__':
    # img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig', (255, 255, 255), 4)
    # vis.vis_cls_log(r'C:\Users\ChengXi\Desktop\cstnet2\pnet2_geomloss.txt')
    # vis.vis_step_cloud(r'C:\Users\ChengXi\Desktop\apart.STEP', color=[38,40,46], n_points=700)
    step_proc.test()

    pass






