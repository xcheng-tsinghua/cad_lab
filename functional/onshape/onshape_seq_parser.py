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
from functional.onshape.OnshapeClient import OnshapeClient
from functional.onshape import topology_parser
import matplotlib.pyplot as plt
from functional.onshape import macro
from functional.onshape.OspGeomBase import point_list_to_numpy
from functional.onshape.OperationParser import Extrude, Revolve, Sweep, Loft
import json
from functional import brep
from OCC.Display.SimpleGui import init_display


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


def show_entity_ids(entity_ids, face_id_dict, edge_id_dict, vert_id_dict):
    """
    显示一组实体 id
    :param entity_ids:
    :param face_id_dict:
    :param edge_id_dict:
    :param vert_id_dict:
    :return:
    """
    # 获取显示数据
    sample_points = []
    for entity_id in entity_ids:
        # 是面的情况
        if entity_id in face_id_dict:
            face_entity = face_id_dict[entity_id]
            for edge_entity in face_entity.edges:
                sample_list = edge_entity.sample()
                sample_points.append(point_list_to_numpy(sample_list))

        elif entity_id in edge_id_dict:
            prim = edge_id_dict[entity_id]
            sample_list = prim.sample()
            sample_points.append(point_list_to_numpy(sample_list))

        elif entity_id in vert_id_dict:
            print('vertex entity not shown')

        else:
            raise ValueError(f'required entity id: {entity_id} not in any topo dict.')

    plot_3d_sketch(sample_points)


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


def get_operation_entities(feat_ofs):
    """
    获取全部建模操作实体
    :param feat_ofs:
    :return:
    """
    operation_entity_all = []
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

        operation_entity_all.append(operation_entity)

    return operation_entity_all


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


def test_parse_bspline_face():
    topo_ofs_file = os.path.join(macro.SAVE_ROOT, 'operation_topo_rollback_14.json')
    with open(topo_ofs_file, 'r') as f:
        entity_topo = json.load(f)

    all_parsed_face = []
    topo_parsed_all = {'regions': [], 'bodies': [], 'faces': [], 'edges': [], 'vertices': []}
    val1st_ofs = entity_topo['result']['message']['value']
    for val1st_item_ofs in val1st_ofs:
        topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

        topo_parsed_all['regions'].extend(topo_parsed['regions'])
        topo_parsed_all['bodies'].extend(topo_parsed['bodies'])
        topo_parsed_all['faces'].extend(topo_parsed['faces'])
        topo_parsed_all['edges'].extend(topo_parsed['edges'])
        topo_parsed_all['vertices'].extend(topo_parsed['vertices'])

        for topoface in topo_parsed['faces']:
            if 'approximateBSplineSurface' in topoface.keys():
                all_parsed_face.append(brep.make_bspline_face(topoface['approximateBSplineSurface']))

    with open(os.path.join(macro.SAVE_ROOT, 'test_face_parse.json'), 'w') as f:
        json.dump(topo_parsed_all, f, ensure_ascii=False, indent=4)

    display, start_display, _, _ = init_display()
    display.DisplayShape(all_parsed_face, update=True)
    start_display()


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
    feat_ofs = onshape_client.request_features(model_url, is_load_ofs, os.path.join(save_root, 'feat_ofs.json'))

    all_feat_id, all_feat_type = get_feat_id(feat_ofs)
    print(f'number of all feat: {len(all_feat_id)}.')

    # 获取全部拓扑
    topo_parsed_all = {'regions': [], 'bodies': [], 'faces': [], 'edges': [], 'vertices': []}
    seq_face = []

    # 全部已解析到的 entity id
    parsed_entity_id_all = []

    request_feat_id = []
    for idx, (feat_id, feat_type) in enumerate(zip(all_feat_id, all_feat_type)):
        # 对于草图则可以合并到后面的建模操作一起请求，因为草图构建的实体不会被更改
        request_feat_id.append(feat_id)

        # 需要回滚到该特征构建之后
        roll_back_idx = idx + 1

        # if feat_type in ('extrude', 'revolve', 'sweep', 'loft', 'fillet', 'chamfer', 'linearPattern', 'circularPattern', 'draft', 'rib', 'mirror'):
        # 向服务器请求当前操作下的模型拓扑
        # 感觉最好还是一个操作请求一次，否则可能遗漏信息，相比节省request，还是弄得完全些
        entity_topo = onshape_client.request_topo_roll_back_to(model_url, request_feat_id, roll_back_idx, is_load_topo, os.path.join(save_root, f'operation_topo_rollback_{idx + 1}.json'))
        parsed_entity_id_all.extend(extract_entity_ids(entity_topo))

        val1st_ofs = entity_topo['result']['message']['value']
        for val1st_item_ofs in val1st_ofs:
            topo_parsed = topology_parser.parse_feat_topo(val1st_item_ofs['message']['value'])

            topo_parsed_all['regions'].extend(topo_parsed['regions'])
            topo_parsed_all['bodies'].extend(topo_parsed['bodies'])
            topo_parsed_all['faces'].extend(topo_parsed['faces'])
            topo_parsed_all['edges'].extend(topo_parsed['edges'])
            topo_parsed_all['vertices'].extend(topo_parsed['vertices'])

            seq_face.append(topo_parsed['faces'])

    # 获取全部的建模操作参数
    operation_entities = get_operation_entities(feat_ofs)

    # 获取未获取但需要的实体 id
    entity_ids_required = []
    for operation_entity in operation_entities:
        entity_ids_required.extend(operation_entity.required_geo)

    not_parsed = in_a_not_in_b(entity_ids_required, parsed_entity_id_all)
    print(f'not obtained topo ids: ', not_parsed)

    # 获取全部角点
    vert_dict = topology_parser.parse_vert_dict(topo_parsed_all)

    # 获取全部边
    edge_dict = topology_parser.parse_edge_dict(topo_parsed_all, vert_dict)

    # 获取全部区域
    face_dict = topology_parser.parse_face_dict(topo_parsed_all, edge_dict)

    # 显示各建模操作的所需元素
    for operation_entity in operation_entities:
        show_entity_ids(operation_entity.required_geo, face_dict, edge_dict, vert_dict)





