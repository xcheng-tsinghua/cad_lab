"""
解析从 onshape 请求回来的模型数据

ofs: onshape feature sequence 表示原始的从 onshape 上下载的文件内容的一部分
Osp: onshape sequence parser 简写
info: 从json中初步解析到的信息，还需进一步处理成对象才可使用

注意，即使是同一 id 的实体，在不同建模步骤下的状态可能也不同
"""
import os
from onshape.OnshapeClient import OnshapeClient
from onshape import topology_parser
from onshape import on_utils
import matplotlib.pyplot as plt
from onshape import macro
from onshape.OspGeomBase import point_list_to_numpy, OspPoint
from onshape import OperationParser
from colorama import Fore, Style

import json


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


def show_entity_ids(entity_ids, entities_all):
    """
    显示一组实体 id
    :param entity_ids:
    :param entities_all:
    :return:
    """
    # 获取显示数据
    sample_points = []
    for entity_id in entity_ids:

        entity = entities_all[entity_id]
        entity_sample_list = entity.sample()

        if isinstance(entity_sample_list[0], list):
            sample_points.extend(entity_sample_list)

        elif isinstance(entity_sample_list[0], OspPoint):
            sample_points.append(entity_sample_list)

        else:
            raise NotImplementedError

    points_numpy = []
    for item in sample_points:
        points_numpy.append(point_list_to_numpy(item))

    plot_3d_sketch(points_numpy)


# def get_all_entity_ids(ofs, result=None):
#     if result is None:
#         result = []
#
#     if isinstance(ofs, dict):
#         for k, v in ofs.items():
#             if k == "geometryIds" and isinstance(v, list):
#                 result.extend(v)
#             else:
#                 get_all_entity_ids(v, result)
#
#     elif isinstance(ofs, list):
#         for item in ofs:
#             get_all_entity_ids(item, result)
#
#     return result
#
#
# def extract_entity_ids(node, out=None):
#     if out is None:
#         out = set()
#
#     if isinstance(node, dict):
#         type_name = node.get("typeName")
#
#         # 关键：MapEntry
#         if type_name == "BTFSValueMapEntry":
#             key = node.get("message", {}).get("key", {})
#             val = node.get("message", {}).get("value", {})
#
#             if (
#                 key.get("typeName") == "BTFSValueString"
#                 and key.get("message", {}).get("value") == "id"
#                 and val.get("typeName") == "BTFSValueString"
#             ):
#                 out.add(val["message"]["value"])
#
#         # 递归所有字段
#         for v in node.values():
#             extract_entity_ids(v, out)
#
#     elif isinstance(node, list):
#         for item in node:
#             extract_entity_ids(item, out)
#
#     return out


def get_operation_cmds(feat_ofs):
    """
    获取全部建模操作实体
    :param feat_ofs:
    :return:
    """
    operation_cmd_all = []
    for fea_item_ofs in feat_ofs['features']:
        fea_type = fea_item_ofs['message']['featureType']

        if fea_type == 'extrude':  # 拉伸
            operation_cmd = OperationParser.Extrude.from_ofs(fea_item_ofs)

        elif fea_type == 'revolve':  # 旋转
            operation_cmd = OperationParser.Revolve.from_ofs(fea_item_ofs)

        elif fea_type == 'sweep':  # 扫描
            operation_cmd = OperationParser.Sweep.from_ofs(fea_item_ofs)

        elif fea_type == 'loft':  # 放样
            operation_cmd = OperationParser.Loft.from_ofs(fea_item_ofs)

        elif fea_type == 'fillet':
            operation_cmd = OperationParser.Fillet.from_ofs(fea_item_ofs)

        elif fea_type == 'chamfer':
            operation_cmd = OperationParser.Chamfer.from_ofs(fea_item_ofs)

        elif fea_type == 'linearPattern':  # 线性阵列
            operation_cmd = OperationParser.LinearPattern.from_ofs(fea_item_ofs)

        elif fea_type == 'circularPattern':  # 圆周阵列
            operation_cmd = OperationParser.CircularPattern.from_ofs(fea_item_ofs)

        elif fea_type == 'draft':  # 拔模
            operation_cmd = OperationParser.Draft.from_ofs(fea_item_ofs)

        elif fea_type == 'rib':  # 拔模
            operation_cmd = OperationParser.Rib.from_ofs(fea_item_ofs)

        elif fea_type == 'mirror':  # 拔模
            operation_cmd = OperationParser.Mirror.from_ofs(fea_item_ofs)

        elif fea_type in ('newSketch', 'cPlane'):  # 新建草图，构建参考面，不产生实体，无需构建建模步骤
            operation_cmd = OperationParser.Mirror.from_ofs(fea_item_ofs)

        else:  # 其它未考虑到的建模步骤
            print(Fore.RED + f'not considered operation type: {fea_type}, save directively' + Style.RESET_ALL)
            operation_cmd = fea_item_ofs

        operation_cmd_all.append(operation_cmd)

    return operation_cmd_all


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


def test():
    topo_ofs_file = os.path.join(macro.SAVE_ROOT, 'operation_topo_rollback_14.json')
    with open(topo_ofs_file, 'r') as f:
        entity_topo = json.load(f)

    topo_parsed_all = {'regions': [], 'bodies': [], 'faces': [], 'edges': [], 'vertices': []}
    val1st_ofs = entity_topo['result']['message']['value']
    for val1st_item_ofs in val1st_ofs:
        topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

        topo_parsed_all['regions'].extend(topo_parsed['regions'])
        topo_parsed_all['bodies'].extend(topo_parsed['bodies'])
        topo_parsed_all['faces'].extend(topo_parsed['faces'])
        topo_parsed_all['edges'].extend(topo_parsed['edges'])
        topo_parsed_all['vertices'].extend(topo_parsed['vertices'])

    # 获取全部拓扑信息
    vert_dict, edge_dict, face_dict, body_dict = topology_parser.parse_topo_dict(topo_parsed_all)

    for _, osp_body in body_dict.items():
        osp_body.show()

    on_utils.show_osp_face_dict(face_dict)


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
    # 构建 onshape api 客户端
    onshape_client = OnshapeClient()

    # 获取最初的操作特征列表
    feat_ofs = onshape_client.request_features(
        model_url,
        is_load_ofs,
        os.path.join(save_root, 'feat_ofs.json')
    )

    all_feat_id, all_feat_type = get_feat_id(feat_ofs)
    print(f'number of all feat: {len(all_feat_id)}.')

    # 获取全部的建模操作参数
    operation_cmd_list = get_operation_cmds(feat_ofs)

    # 获取的全部拓扑
    entities_all = vert_ids_all = edge_ids_all = face_ids_all = body_ids_all = None

    request_feat_id = []
    for idx, (feat_id, feat_type, operation_cmd) in enumerate(zip(all_feat_id, all_feat_type, operation_cmd_list)):
        # 对于草图则可以合并到后面的建模操作一起请求，因为草图构建的实体不会被更改
        request_feat_id.append(feat_id)

        # 需要回滚到该特征构建之后
        roll_back_idx = idx + 1

        # 向服务器请求当前操作下的模型拓扑
        # 感觉最好还是一个操作请求一次，否则可能遗漏信息，相比节省request，还是弄得完全些
        entity_topo = onshape_client.request_topo_roll_back_to(
            model_url,
            request_feat_id,
            roll_back_idx,
            is_load_topo,
            os.path.join(save_root, f'operation_topo_rollback_{roll_back_idx}.json')
        )

        # 截止到当前操作步骤获得的拓扑实体
        topo_parsed_now = {'bodies': [], 'faces': [], 'edges': [], 'vertices': []}

        val1st_ofs = entity_topo['result']['message']['value']
        for val1st_item_ofs in val1st_ofs:
            topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

            # topo_parsed_step['regions'].extend(topo_parsed['regions'])
            topo_parsed_now['bodies'].extend(topo_parsed['bodies'])
            topo_parsed_now['faces'].extend(topo_parsed['faces'])
            topo_parsed_now['edges'].extend(topo_parsed['edges'])
            topo_parsed_now['vertices'].extend(topo_parsed['vertices'])

        # 解析本次获取的全部实体
        entities, vert_ids, edge_ids, face_ids, body_ids = topology_parser.parse_topo_dict(topo_parsed_now)

        # 解析到的拓扑实体需要合并前一步的拓扑实体，因为本次使用的拓扑实体可能是前面的建模步骤创建的
        # 如果解析到前面已解析的拓扑实体，本次依赖的拓扑实体以新的为准
        if idx == 0:
            entities_all = entities
            vert_ids_all = vert_ids
            edge_ids_all = edge_ids
            face_ids_all = face_ids
            body_ids_all = body_ids

        else:
            entities_all.update(entities)
            vert_ids_all.update(vert_ids)
            edge_ids_all.update(edge_ids)
            face_ids_all.update(face_ids)
            body_ids_all.update(body_ids)

        # 显示实体
        if feat_type in ('extrude', 'revolve', 'sweep', 'loft',
                         'fillet', 'chamfer',
                         'linearPattern', 'circularPattern',
                         'draft', 'rib', 'mirror'
                         ):
            # 校验是否有操作需要但是未获取到的 id
            not_parsed = in_a_not_in_b(operation_cmd.required_geo, list(entities_all.keys()))
            if not_parsed:
                print(Fore.RED + f'not parsed entity ids: {not_parsed} / {roll_back_idx} / {feat_type}' + Style.RESET_ALL)
                raise ValueError
            else:
                print(Fore.GREEN + f'all required entity ids are already parsed' + Style.RESET_ALL)
                # show_entity_ids(operation_cmd.required_geo, entities_all)

            # for body_id in body_ids:
            #     entities_all[body_id].show()





