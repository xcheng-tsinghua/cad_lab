import vis
import step_proc
import img_proc


if __name__ == '__main__':
    # file_name = ['D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000000\\00000000_290a9120f9f249a7a05cfe9c_step_000.step', 'D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000001\\00000001_1ffb81a71e5b402e966b9341_step_000.step', 'D:\\document\\DeepLearning\\tmp\\STEPMillion_pack1\\00000002\\00000002_1ffb81a71e5b402e966b9341_step_001.step']
    #
    # for cstep in file_name:
    #     vis.vis_step_file(cstep)

    # vis.vis_step_file(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00000038\00000038_c7d977f326364e35bb5b5d27_step_001.step')

    # img_proc.remove_png_white_pixel(r'E:\document\deeplearning_idea\参数化点云\参数化点云小论文\网站\cstnetwork.io\static\images\data_percentage.png')
    img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\fig')

    # img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\program')

    # img_proc.remove_png_white_pixel_batched(r'C:\Users\ChengXi\Desktop\valids\v2')

    # 简单r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00000029\00000029_ad34a3f60c4a4caa99646600_step_009.step'

    # vis.vis_data2d('log/loss-time.txt')

    # adict = {'Geom_Plane': 1745610, 'Geom_CylindricalSurface': 439119, 'Geom_BSplineSurface': 294139, 'Geom_RectangularTrimmedSurface': 271824, 'Geom_ToroidalSurface': 64732, 'Geom_ConicalSurface': 43831, 'Geom_SurfaceOfLinearExtrusion': 43086, 'Geom_SphericalSurface': 23826, 'Geom_SurfaceOfRevolution': 20460, 'Geom_OffsetSurface': 2490}
    #
    # all_ins = 0
    #
    # for c_key in adict.keys():
    #     all_ins += adict[c_key]
    #
    # print(f'instance_all: {all_ins}')
    #
    # plane_all = adict['Geom_Plane']
    # cylinder_all = adict['Geom_CylindricalSurface']
    # cone_all = adict['Geom_ConicalSurface'] * 4
    # other_all = adict['Geom_BSplineSurface'] // 2
    #
    # all_ins = plane_all + cylinder_all + cone_all + other_all
    #
    # print(f'plane_count: {plane_all / all_ins}')
    # print(f'cylinder_count: {cylinder_all / all_ins}')
    # print(f'cone_count: {cone_all / all_ins}')
    # print(f'other_count: {other_all / all_ins}')


