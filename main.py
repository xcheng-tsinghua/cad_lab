import utils
import vis
import step_proc
from tqdm import tqdm


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # shape = step_proc.step_read_ocaf(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step')
    # vis.vis_shapeocc(shape)

    # aprtlist = step_proc.assemble_explode(r'C:\Users\ChengXi\Desktop\装配体1.STEP')
    # aprtlist = step_proc.assemble_explode(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step')
    #
    # for c_shape in aprtlist:
    #     vis.vis_shapeocc(c_shape)

    aroot = r'D:\document\DeepLearning\tmp\STEPMillion_pack1'
    files_all = utils.get_allfiles(aroot, 'step')
    assem = 0
    assem_all = []
    for c_file in tqdm(files_all, total=len(files_all)):
        if step_proc.assembly_filter(c_file):
            assem += 1
            assem_all.append(c_file)
            print(c_file)

    print(assem)

    with open('log/assems.txt', 'w') as f:
        for cline in assem_all:
            f.write(cline + '\n')

    # print(step_proc.assembly_filter(r'C:\Users\ChengXi\Desktop\gear-paper.STEP'))
    # print(step_proc.assembly_filter(r'D:\document\DeepLearning\tmp\STEPMillion_pack1\00003236\00003236_64bf0eb2d82b4b9aac59e530_step_000.step'))



