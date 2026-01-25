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


def get_all_entity_ids(ofs, result=None):
    if result is None:
        result = []

    if isinstance(ofs, dict):
        for k, v in ofs.items():
            if k == "geometryIds" and isinstance(v, list):
                result.extend(v)
            else:
                get_all_entity_ids(v, result)

    elif isinstance(ofs, list):
        for item in ofs:
            get_all_entity_ids(item, result)

    return result


def extract_entity_ids(node, out=None):
    if out is None:
        out = set()

    if isinstance(node, dict):
        type_name = node.get("typeName")

        # 关键：MapEntry
        if type_name == "BTFSValueMapEntry":
            key = node.get("message", {}).get("key", {})
            val = node.get("message", {}).get("value", {})

            if (
                key.get("typeName") == "BTFSValueString"
                and key.get("message", {}).get("value") == "id"
                and val.get("typeName") == "BTFSValueString"
            ):
                out.add(val["message"]["value"])

        # 递归所有字段
        for v in node.values():
            extract_entity_ids(v, out)

    elif isinstance(node, list):
        for item in node:
            extract_entity_ids(item, out)

    return out


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
        ofs = onshape_client.request_features(model_url)

        print('保存原始特征列表')
        with open(ofs_path, 'w') as f:
            json.dump(ofs, f, ensure_ascii=False, indent=4)

    # # 获取全部的草图和建模操作的 id
    # all_feat_id = []
    # for i, fea_item_ofs in enumerate(ofs['features']):
    #     feat_type = fea_item_ofs['message']['featureType']
    #     feat_id = fea_item_ofs['message']['featureId']
    #
    #     print(f'feat_type: {feat_type}, feat_id: {feat_id}')
    #     all_feat_id.append(fea_item_ofs['message']['featureId'])
    #
    # # 获取全部的草图和建模操作产生的实体的 topology
    # topo_path = os.path.join(save_root, 'sketch_operation_topo.json')
    # if is_load_topo:
    #     print(f'从文件加载原始拓扑列表: {topo_path}')
    #     with open(topo_path, 'r') as f:
    #         sketch_operation_topo = json.load(f)
    # else:
    #     print('从 onshape 请求原始拓扑列表')
    #     sketch_operation_topo = onshape_client.request_multi_feat_topology(model_url, all_feat_id)
    #
    #     print('保存原始拓扑列表')
    #     with open(topo_path, 'w') as f:
    #         json.dump(sketch_operation_topo, f, ensure_ascii=False, indent=4)

    # 获取全部需要的实体 id
    entity_ids_all = get_all_entity_ids(ofs)

    # 获取全部的实体 topo
    topo_path = os.path.join(save_root, 'entity_topo.json')
    if is_load_topo:
        print(f'从文件加载原始拓扑列表: {topo_path}')
        with open(topo_path, 'r') as f:
            sketch_operation_topo = json.load(f)
    else:
        print('从 onshape 请求原始拓扑列表')
        sketch_operation_topo = onshape_client.request_multi_entity_topology(model_url, entity_ids_all)

        print('保存原始拓扑列表')
        with open(topo_path, 'w') as f:
            json.dump(sketch_operation_topo, f, ensure_ascii=False, indent=4)

    all_parsed_ids = extract_entity_ids(sketch_operation_topo)

    a = set(all_parsed_ids)
    b = set(entity_ids_all)
    intersection = list(a & b)  # 交集
    union = list(a | b)  # 并集
    diff_a = list(a - b)
    diff_b = list(b - a)

    val1st_ofs = sketch_operation_topo['result']['message']['value']

    # 获取全部拓扑
    all_topo_parsed = {'faces': [], 'regions': [], 'edges': [], 'vertices': []}
    for val1st_item_ofs in val1st_ofs:
        topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

        all_topo_parsed['faces'].extend(topo_parsed['faces'])
        # all_topo_parsed['regions'].extend(topo_parsed['regions'])
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



