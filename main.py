import vis
import step_proc


def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 Ctrl+F8 切换断点。


# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    # shape = step_proc.step_read_ocaf(r'C:\Users\ChengXi\Desktop\gear-paper.STEP')
    # vis.vis_shapeocc(shape)

    aprtlist = step_proc.assemble_explode(r'C:\Users\ChengXi\Desktop\装配体1.STEP')
    vis.vis_shapeocc(aprtlist[0])
    vis.vis_shapeocc(aprtlist[1])

    # print(step_proc.assembly_filter(r'C:\Users\ChengXi\Desktop\gear-paper.STEP'))
    # print(step_proc.assembly_filter(r'C:\Users\ChengXi\Desktop\装配体1.STEP'))
