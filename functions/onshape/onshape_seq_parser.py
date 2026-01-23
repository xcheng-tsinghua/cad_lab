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
from functions.onshape import topology_parser
import matplotlib.pyplot as plt
from functions.onshape import macro
from functions.onshape.OspGeomBase import OspPoint
import numpy as np


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

    orig_ofs = client.request_features(did, wid, eid).json()
    return orig_ofs


def request_multi_feat_topology(
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
    res = onshape_client.request_multi_feat_topology(did, wid, eid, all_sketch_id)
    return res


def parse_onshape_topology(
        model_url: str = macro.URL,
        is_load_ofs: bool = True,
        is_load_topo: bool = True,
        save_root: str = macro.SAVE_ROOT
):
    """
    解析全部草图和建模操作的拓扑
    :return:
    """
    onshape_client = OnshapeClient()

    ofs_path = os.path.join(save_root, 'orig_ofs.json')
    if is_load_ofs:
        print(f'从文件加载原始特征列表: {ofs_path}')
        with open(ofs_path, 'r') as f:
            ofs = json.load(f)
    else:
        print('从 onshape 请求原始特征列表')
        ofs = request_model_ofs(onshape_client, model_url)

        print('保存原始特征列表')
        with open(ofs_path, 'w') as f:
            json.dump(ofs, f, ensure_ascii=False, indent=4)

    # 获取全部的草图和建模操作的 id
    all_feat_id = []
    for i, fea_item_ofs in enumerate(ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']
        feat_id = fea_item_ofs['message']['featureId']

        print(f'feat_type: {feat_type}, feat_id: {feat_id}')
        all_feat_id.append(fea_item_ofs['message']['featureId'])

    # 获取全部的草图和建模操作产生的实体的 topology
    topo_path = os.path.join(save_root, 'sketch_operation_topo.json')
    if is_load_topo:
        print(f'从文件加载原始拓扑列表: {topo_path}')
        with open(topo_path, 'r') as f:
            sketch_operation_topo = json.load(f)
    else:
        print('从 onshape 请求原始拓扑列表')
        sketch_operation_topo = request_multi_feat_topology(onshape_client, model_url, all_feat_id)

        print('保存原始拓扑列表')
        with open(topo_path, 'w') as f:
            json.dump(sketch_operation_topo, f, ensure_ascii=False, indent=4)

    val1st_ofs = sketch_operation_topo['result']['message']['value']

    # 获取全部拓扑
    all_topo_parsed = {'regions': [], 'edges': [], 'vertices': []}
    for val1st_item_ofs in val1st_ofs:
        topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

        all_topo_parsed['regions'].extend(topo_parsed['regions'])
        all_topo_parsed['edges'].extend(topo_parsed['edges'])
        all_topo_parsed['vertices'].extend(topo_parsed['vertices'])

    # 获取全部角点
    vert_dict = topology_parser.parse_vert_dict(all_topo_parsed)

    # 获取全部边
    edge_dict = topology_parser.parse_edge_dict(all_topo_parsed, vert_dict)

    # 获取全部区域
    region_dict = topology_parser.parse_region_dict(all_topo_parsed, edge_dict)

    # 获取绘图元素：
    all_plots = []
    for _, region in region_dict.items():
        for prim in region.primitive_list:
            sample_list = prim.sample()
            all_plots.append(point_list_to_numpy(sample_list))

    # 绘图
    plot_3d_sketch(all_plots)



