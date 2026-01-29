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
from functions.onshape.OnshapeClient import OnshapeClient
from functions.onshape import topology_parser
import matplotlib.pyplot as plt
from functions.onshape import macro
from functions.onshape.OspGeomBase import point_list_to_numpy
from functions.onshape.OperationParser import Extrude, Revolve, Sweep, Loft


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


def get_required_entity_id(feat_ofs):
    """
    获取全部所需的实体 id美丽如 JDC
    :param feat_ofs:
    :return:
    """
    operation_entity_all = []
    required_geo_list = []
    for fea_item_ofs in feat_ofs['features']:
        fea_type = fea_item_ofs['message']['featureType']

        if fea_type == 'newSketch':  # 新草图
            continue

        elif fea_type == 'extrude':  # 拉伸
            operation_entity = Extrude.from_ofs(fea_item_ofs)

        elif fea_type == 'revolve':  # 旋转
            operation_entity = Revolve.from_ofs(fea_item_ofs)

        elif fea_type == 'sweep':  # 扫描
            operation_entity = Sweep.from_ofs(fea_item_ofs)

        elif fea_type == 'loft':  # 放样
            operation_entity = Loft.from_ofs(fea_item_ofs)

        else:    # 圆角、倒角、阵列 ？
            continue

        required_geo_list.extend(operation_entity.required_geo)
        operation_entity_all.append(operation_entity)

    entity_ids_required = list(set(required_geo_list))
    return operation_entity_all, entity_ids_required


def get_feat_id(feat_ofs):
    """
    获得全部建模命令的 id，例如草图、拉伸的 id: FcnjNG48IKcdqOW_0
    :param feat_ofs:
    :return:
    """
    all_feat_id = []
    all_feat_type = []
    for i, fea_item_ofs in enumerate(feat_ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']
        feat_id = fea_item_ofs['message']['featureId']

        print(f'feat_type: {feat_type}, feat_id: {feat_id}')

        # if feat_type in ('extrude', 'revolve', 'sweep', 'loft'):
        all_feat_id.append(fea_item_ofs['message']['featureId'])

        all_feat_type.append(feat_type)

    return all_feat_id, all_feat_type


def in_a_not_in_b(a, b):
    """
    找出在数组 a 中存在但不在 b 中存在的值
    :param a:
    :param b:
    :return:
    """
    a = set(a)
    b = set(b)
    diff_a = list(a - b)
    return diff_a


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

    # 获取最初的操作特征列表
    feat_ofs = onshape_client.request_features(model_url, is_load_ofs, os.path.join(save_root, 'feat_ofs.json'))
    # all_feat_id, all_feat_type =

    # 获取全部拓扑
    all_topo_parsed = {'faces': [], 'regions': [], 'edges': [], 'vertices': []}
    seq_face = []

    # 全部已解析到的 entity id
    parsed_entity_id_all = []

    request_feat_id = []
    request_times = 0

    all_feat_id, all_feat_type = get_feat_id(feat_ofs)
    print(f'number of all feat: {len(all_feat_id)}.')

    for idx, (feat_id, feat_type) in enumerate(zip(all_feat_id, all_feat_type)):
    # for idx in enumerate(zip(*get_feat_id(feat_ofs))):

        # 对于草图则可以合并到后面的建模操作一起请求，因为草图构建的实体不会被更改
        request_feat_id.append(feat_id)

        if feat_type in ('extrude', 'revolve', 'sweep', 'loft', '倒角、圆角、阵列？'):
            # 向服务器请求当前操作下的模型拓扑
            entity_topo = onshape_client.request_topo_roll_back_to(model_url, request_feat_id, idx + 1, is_load_topo, os.path.join(save_root, f'operation_topo_rollback_{request_times}.json'))

            # print(f'processed feat number：{len(request_feat_id)}')

            parsed_entity_id_all.extend(get_all_entity_ids(entity_topo))

            val1st_ofs = entity_topo['result']['message']['value']
            for val1st_item_ofs in val1st_ofs:
                topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

                all_topo_parsed['faces'].extend(topo_parsed['faces'])
                all_topo_parsed['edges'].extend(topo_parsed['edges'])
                all_topo_parsed['vertices'].extend(topo_parsed['vertices'])

                seq_face.append(topo_parsed['faces'])

                # if feat_type in ('extrude', 'revolve', 'sweep', 'loft'):
                #     seq_face.append(topo_parsed['faces'])

            # 清空需请求的id
            request_times += 1
            # request_feat_id.clear()

    required_entity_id = get_required_entity_id(feat_ofs)[1]

    not_parsed = in_a_not_in_b(required_entity_id, parsed_entity_id_all)


    # 获取逐步获得的几何体列表
    # entity_topo = onshape_client.request_multi_feat_topology(model_url, all_feat_id[: 2], is_load_topo, os.path.join(save_root, 'operation_topo.json'))
    # entity_topo = onshape_client.request_topo_roll_back_to(model_url, all_feat_id[: 2], 2, is_load_topo,
    #                                                          os.path.join(save_root, 'operation_topo_rollback2.json'))

    # 获取全部拓扑
    # all_topo_parsed = {'faces': [], 'regions': [], 'edges': [], 'vertices': []}
    # seq_face = []
    # val1st_ofs = entity_topo['result']['message']['value']
    # for feat_type, val1st_item_ofs in zip(all_feat_type, val1st_ofs):
    #     topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])
    #
    #     all_topo_parsed['faces'].extend(topo_parsed['faces'])
    #     all_topo_parsed['edges'].extend(topo_parsed['edges'])
    #     all_topo_parsed['vertices'].extend(topo_parsed['vertices'])
    #
    #     if feat_type in ('extrude', 'revolve', 'sweep', 'loft'):
    #         seq_face.append(topo_parsed['faces'])

    # 获取全部角点
    vert_dict = topology_parser.parse_vert_dict(all_topo_parsed)

    # 获取全部边
    edge_dict = topology_parser.parse_edge_dict(all_topo_parsed, vert_dict)

    # 获取全部区域
    face_dict = topology_parser.parse_region_dict(all_topo_parsed, edge_dict)



    # 获取绘图元素：
    # all_plots = []
    # for _, region in face_dict.items():
    #     for prim in region.primitive_list:
    #         sample_list = prim.sample()
    #         all_plots.append(point_list_to_numpy(sample_list))

    # for _, prim in edge_dict.items():
    #     sample_list = prim.sample()
    #     all_plots.append(point_list_to_numpy(sample_list))

    for seq_face_item in seq_face:
        all_plots = []

        for c_face in seq_face_item:

            for edge_id in c_face['edges']:
                prim = edge_dict[edge_id]
                sample_list = prim.sample()
                all_plots.append(point_list_to_numpy(sample_list))

        plot_3d_sketch(all_plots)

    # 绘图
    # plot_3d_sketch(all_plots)



