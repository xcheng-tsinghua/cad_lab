"""
登录onshape的配置文件：
./config/onshape_credit.json

文件内容示例：
{
  "onshape_url": "https://cad.onshape.com",
  "access_key": "your_access_key",
  "secret_key": "your_secret_key"
}

ofs: onshape feature sequence 表示原始的从 onshape 上下载的文件内容的一部分
Osp: onshape sequence parser 简写

"""
import os
import json
from functions.onshape.OnshapeClient import OnshapeClient
from functions.onshape.SketchParser import Sketch
import matplotlib.pyplot as plt
from functions.onshape import macro
from functions.onshape.OspGeomBase import OspPoint
import numpy as np
from datetime import datetime


def point_list_to_numpy(osp_point_list: list[OspPoint]):
    """
    将一个包含 n 个点的列表转换为 [n * 3] 的 numpy 数组
    :param osp_point_list:
    :return:
    """
    np_list = []
    for c_osp in osp_point_list:
        np_list.append(c_osp.to_numpy())

    np_list = np.vstack(np_list)
    return np_list


def plot_3d_sketch(sample_list):
    # 创建图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    for sample in sample_list:
        ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], label='3D Sketch', color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def request_sketch_plane(skh_parser_list, save_path, model_url):
    """
    批量获取草图平面信息
    :param skh_parser_list:
    :param save_path:
    :param model_url:
    :return:
    """
    v_list = model_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    onshape_client = OnshapeClient()

    all_req_id = []
    for i, item in enumerate(skh_parser_list):
        all_req_id.append(item.sketch_plane_id)

    res = onshape_client.get_face_by_id(did, wid, eid, all_req_id)

    with open(save_path, 'w') as f:
        json.dump(res.json(), f, ensure_ascii=False, indent=4)

    return res.json()


def request_model_ofs(
        client: OnshapeClient,
        model_url: str
):
    """
    从服务器请求原始特征列表
    :param client:
    :param model_url:
    :return:
    """
    v_list = model_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    orig_ofs = client.get_features(did, wid, eid).json()
    return orig_ofs


def request_sketch_topology(
        onshape_client: OnshapeClient,
        model_url: str,
        all_sketch_id: list[str]
):
    """
    批量获取草图中的全部区域信息
    :param onshape_client:
    :param model_url:
    :param all_sketch_id: 草图节点 ID 列表
    :return:
    """
    v_list = model_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    # 一次请求全部的 sketch topo
    res = onshape_client.eval_multi_sketch_topology(did, wid, eid, all_sketch_id)
    return res


def parse_onshape_topology(is_load_ofs_and_topo=False):
    """
    解析全部草图和建模操作的拓扑
    :return:
    """
    onshape_client = OnshapeClient()
    model_url = macro.URL
    save_root = macro.SAVE_ROOT

    ofs_path = os.path.join(save_root, 'orig_ofs.json')

    if is_load_ofs_and_topo:
        print('从文件加载原始特征列表')
        with open(ofs_path, 'r') as f:
            ofs = json.load(f)

    else:
        print('从服务器请求原始特征列表')
        ofs = request_model_ofs(onshape_client, model_url)

        print('保存原始特征列表')
        with open(ofs_path, 'w') as f:
            json.dump(ofs, f, ensure_ascii=False, indent=4)

    # 获取全部的草图和建模操作的 id
    all_sketch_id = []
    all_operation_id = []
    for i, fea_item_ofs in enumerate(ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']

        if feat_type == 'newSketch':
            all_sketch_id.append(fea_item_ofs['message']['featureId'])

        elif feat_type in ('extrude', 'revolve', 'loft', 'sweep', '倒角、圆角、线性阵列、圆周阵列？'):
            all_operation_id.append(fea_item_ofs['message']['featureId'])

    # 获取全部的草图和建模操作产生的实体的 topology
    topo_path = os.path.join(save_root, 'sketch_operation_topo.json')
    if is_load_ofs_and_topo:
        print('从文件加载原始拓扑列表')
        with open(topo_path, 'r') as f:
            sketch_operation_topo = json.load(f)

    else:
        sketch_operation_topo = request_sketch_topology(onshape_client, macro.URL, all_sketch_id + all_operation_id)

        print('保存原始拓扑列表')
        with open(topo_path, 'w') as f:
            json.dump(sketch_operation_topo, f, ensure_ascii=False, indent=4)

    val1st_ofs = sketch_operation_topo['result']['message']['value']

    # 将草图和操作形成的拓扑分开
    # n_sketch = len(all_sketch_id)
    # val1st_sketch_ofs, val1st_operation_ofs = val1st_ofs[:n_sketch], val1st_ofs[n_sketch:]

    # 获取全部拓扑
    all_sketch_parsed = []
    for val1st_item_ofs in val1st_ofs:
        sketch_parsed = Sketch(val1st_item_ofs)
        all_sketch_parsed.append(sketch_parsed)

    # 获取绘图元素：
    all_plots = []
    for sketch in all_sketch_parsed:
        for region in sketch.region_list:
            for prim in region.primitive_list:
                sample_list = prim.sample()
                all_plots.append(point_list_to_numpy(sample_list))

    # 绘图
    plot_3d_sketch(all_plots)
    return all_sketch_parsed


def test_parse_sketch():

    ofs_path = os.path.join(macro.SAVE_ROOT, 'orig_ofs.json')
    with open(ofs_path, 'r') as f:
        ofs = json.load(f)

    all_sketch_id = []
    all_operation_id = []
    for i, fea_item_ofs in enumerate(ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']

        if feat_type == 'newSketch':
            all_sketch_id.append(fea_item_ofs['message']['featureId'])

        elif feat_type in ('extrude', 'revolve', 'loft', 'sweep', '倒角、圆角、线性阵列、圆周阵列？'):
            all_operation_id.append(fea_item_ofs['message']['featureId'])

    # 请求服务器一次获取所有草图区域拓扑信息以及操作的拓扑信息
    # all_sketch_topo = request_sketch_topology(all_sketch_id + all_operation_id, macro.URL)
    # timestamp = datetime.now().strftime('%H_%M_%S')
    # target_path = os.path.join(macro.SAVE_ROOT, f'all_sketch_ops_topo_{timestamp}.json')
    # with open(target_path, 'w') as f:
    #     json.dump(all_sketch_topo, f, ensure_ascii=False, indent=4)
    # print('save all sketch and operation topo succeed!')
    # exit('asasadasdaaa')

    # 解析全部草图参数
    # all_topo_file = os.path.join(macro.SAVE_ROOT, 'all_ops_topo_13_16_29.json')
    all_topo_file = os.path.join(macro.SAVE_ROOT, 'all_sketch_topo_20_16_58.json')
    with open(all_topo_file, 'r') as f:
        all_sketch_topo = json.load(f)

    val1st_ofs = all_sketch_topo['result']['message']['value']

    # 获取每个平面的拓扑
    all_sketch_parsed = []
    for val1st_item_ofs in val1st_ofs:
        sketch_parsed = Sketch(val1st_item_ofs)
        all_sketch_parsed.append(sketch_parsed)

    # 获取绘图元素：
    # all_plots = []
    # for sketch in all_sketch_parsed:
    #     for _, region in sketch.region_dict.items():
    #         for prim in region.primitive_list:
    #             sample_list = prim.sample()
    #             all_plots.append(point_list_to_numpy(sample_list))

    all_plots = []
    for sketch in all_sketch_parsed:
        for _, prim in sketch.edge_dict.items():
            sample_list = prim.sample()
            all_plots.append(point_list_to_numpy(sample_list))

    # 绘图
    plot_3d_sketch(all_plots)
    return all_sketch_parsed


