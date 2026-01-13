import win32com.client
import os
import pythoncom


def solidworks_partsave(dir_path):
    '''
    SolidWorks 模型自动切换配置保存到 STEP
    '''
    # 建立com连接,如只有一个版本,可以只写"sldworks.Application"
    swApp = win32com.client.Dispatch('sldworks.Application')
    # 提升API交互效率
    swApp.CommandInProgress = True
    # 显示Solidworks界面
    swApp.visible = True
    # 打开文件

    part_name = os.listdir(dir_path)

    config_name = []
    for i in range(200):
        config_name.append(f'情形 {i}')

    for part in part_name:
        file_path = os.path.join(dir_path, part)
        # 打开文件
        longstatus = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)
        longwarnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)

        try:
            swApp.OpenDoc6(file_path, 1, 0, '', longstatus, longwarnings)
        except:
            print('无法打开该文件：' + file_path)
            continue

        curr_cls_dir = os.path.join(dir_path, part[:-7])
        os.makedirs(curr_cls_dir, exist_ok=True)

        swModel = swApp.ActiveDoc
        for config in config_name:
            # 激活配置
            boolstatus = swModel.ShowConfiguration2(config)
            if not boolstatus:
                print('跳过配置：', config)
                continue

            print('激活配置：', config)
            # 保存当前文件
            filename = os.path.join(curr_cls_dir, part[:-7] + config + '.STEP').replace('情形', 'case')

            # 错误和警告
            Errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)
            Warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)

            # 除PDF文件外,其余格式SaveAs第四个参数均使用Nothing
            Nothing = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
            boolstatus = swModel.Extension.SaveAs(filename, 0, 0, Nothing, Errors, Warnings)
            swModel.Extension.SelectByID2("D1@草图1@零件1.SLDPRT", "DIMENSION", 2.37343987303689E-02, -1.78145456424655E-02, -0.024302098210038, False, 0, Nothing, 0)
            if boolstatus:
                print('文件另存成功')
            else:
                print(f'文件另存失败,出现如下错误:{Errors}')
                print(f'文件另存失败,出现如下警告:{Warnings}')

        # 关闭文件
        swApp.CloseDoc(file_path)


def solidworks_partconfig_to_step(part_path, save_path):
    """
    将part_path对应的SolidWorks模型的全部配置保存至save_path
    """
    # 文件名
    # 去掉路径
    file_name_with_ext = os.path.basename(part_path)
    file_name = os.path.splitext(file_name_with_ext)[0]

    # 建立com连接,如只有一个版本,可以只写"sldworks.Application"
    swApp = win32com.client.Dispatch('sldworks.Application')
    # 提升API交互效率
    swApp.CommandInProgress = True
    # 显示Solidworks界面
    swApp.visible = True

    # 打开文件
    longstatus = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)
    longwarnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)

    try:
        model = swApp.OpenDoc6(part_path, 1, 0, '', longstatus, longwarnings)
    except:
        raise ValueError('无法打开该文件：' + part_path)

    # 获取配置管理器
    config_name = []
    for i in range(200):
        config_name.append(f'情形 {i}')

    swModel = swApp.ActiveDoc
    for config in config_name:
        # 激活配置
        boolstatus = swModel.ShowConfiguration2(config)
        if not boolstatus:
            print('跳过配置：', config)
            continue

        print('激活配置：', config)
        # 保存当前文件
        filename = os.path.join(save_path, file_name + config + '.STEP').replace('情形', 'case')

        # 错误和警告
        Errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)
        Warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, -1)

        # 除PDF文件外,其余格式SaveAs第四个参数均使用Nothing
        Nothing = win32com.client.VARIANT(pythoncom.VT_DISPATCH, None)
        boolstatus = swModel.Extension.SaveAs(filename, 0, 0, Nothing, Errors, Warnings)
        swModel.Extension.SelectByID2("D1@草图1@零件1.SLDPRT", "DIMENSION", 2.37343987303689E-02, -1.78145456424655E-02,
                                      -0.024302098210038, False, 0, Nothing, 0)
        if boolstatus:
            print('文件另存成功')
        else:
            print(f'文件另存失败,出现如下错误:{Errors}')
            print(f'文件另存失败,出现如下警告:{Warnings}')

    # 关闭文件
    swApp.CloseDoc(part_path)


def get_model_view():
    swApp = win32com.client.Dispatch("SldWorks.Application")
    model = swApp.ActiveDoc

    # 获取当前视角的方向信息
    view = model.ActiveView
    viewOrientation = view.Orientation

    # 输出当前视角信息
    print(f"当前视角方向: {viewOrientation}")


if __name__ == "__main__":
    # solidworks_partsave(r'D:\document\DeepLearning\DataSet\参数化零件模板')
    get_model_view()
    pass









