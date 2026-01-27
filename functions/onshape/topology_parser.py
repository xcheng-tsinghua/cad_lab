"""
解析草图和各种建模命令获得的 区域、边、点 拓扑
"""

from functions.onshape.OspGeomBase import OspPoint, OspCoordSystem
from functions.onshape.OspGeomEdge import OspLine, OspCircle, OspEllipse, OspBSpline, OspRegion
from functions.onshape import utils, macro
from warnings import warn


def parse_vert_dict(sketch_topology):
    """
    将草图拓扑解析为 id 到具体点的映射
    :param sketch_topology:
    :return:
    """
    vertices_topo_dict = {item['id']: OspPoint.from_list(item['param']) for item in sketch_topology['vertices']}
    return vertices_topo_dict


def parse_edge_dict(sketch_topology, vert_dict):
    """
    将草图拓扑解析为 id 到具体边的映射
    :param sketch_topology:
    :param vert_dict:
    :return:
    """
    edge_topo_list = sketch_topology['edges']

    edge_dict = {}
    for edge_topo_item in edge_topo_list:
        edge_id = edge_topo_item['id']
        edge_type = edge_topo_item['param']['curveType']

        # 将对应边解析为几何对象
        if edge_type == 'LINE':
            # 是直线，解析两个点
            edge_points_parsed = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)
            edge_parsed = OspLine(edge_points_parsed[0], edge_points_parsed[1], edge_id)

        elif edge_type == 'CIRCLE':
            # 是圆，解析圆心三维坐标、所在平面法线、半径、始末点

            # 找到所在的局部坐标系
            coord_sys = OspCoordSystem.from_parsed_ofs(edge_topo_item['param']['coordSystem'])

            # 获取半径
            radius = edge_topo_item['param']['radius']

            # 获取端点
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)

            # 获取中点
            midpoint = OspPoint.from_list(edge_topo_item['midpoint'])

            # 构造圆
            edge_parsed = OspCircle(coord_sys, radius, vertices[0], midpoint, vertices[1], edge_id)

        elif edge_type == 'ELLIPSE':
            # 找到所在的局部坐标系
            coord_sys = OspCoordSystem.from_parsed_ofs(edge_topo_item['param']['coordSystem'])

            # 获取半长轴 majorRadius
            major_radius = edge_topo_item['param']['majorRadius']

            # 获取短长轴 minorRadius
            minor_radius = edge_topo_item['param']['minorRadius']

            # 获取端点
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)

            # 获取中点
            midpoint = OspPoint.from_list(edge_topo_item['midpoint'])

            # 构造椭圆
            edge_parsed = OspEllipse(coord_sys, major_radius, minor_radius, vertices[0], midpoint, vertices[1], edge_id)

        elif edge_type == 'SPLINE':
            # 获取控制点坐标
            ctrl_points_raw = edge_topo_item['param']['controlPoints']  # 二重数组，表示一系列点
            ctrl_points = [OspPoint.from_list(point_coord) for point_coord in ctrl_points_raw]

            # 获取次数
            degree = round(edge_topo_item['param']['degree'])

            # 获取 dimension
            dimension = round(edge_topo_item['param']['dimension'])

            # 获取 isPeriodic
            is_periodic = edge_topo_item['param']['isPeriodic']

            # 获取 isRational
            is_rational = edge_topo_item['param']['isRational']

            # 获取 knots
            knots = edge_topo_item['param']['knots']

            # 获取 weights
            weights = edge_topo_item['param']['weights'] if is_rational else None

            # 获取端点
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)

            # 构造自由曲线
            edge_parsed = OspBSpline(ctrl_points, degree, dimension, is_periodic, is_rational, knots, weights, vertices[0], vertices[1], edge_id)

        elif edge_type == 'OTHER':
            # 未知类型，直接解析为两个点
            warn('edge type OTHER occurred, treated as LINE')
            edge_points_parsed = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)
            edge_parsed = OspLine(edge_points_parsed[0], edge_points_parsed[1], edge_id)

        else:
            raise NotImplementedError(f'unsupported edge type: {edge_type}')

        edge_dict[edge_id] = edge_parsed

    return edge_dict


def parse_region_dict(sketch_topology, edge_dict):
    """
    将草图拓扑转化为区域
    :param sketch_topology:
    :param edge_dict:
    :return:
    """
    # 将原始的 topology 解析为区域
    region_dict = {}
    for region_topo_item in sketch_topology['faces']:
        region_edge_list = [edge_dict[edge_id] for edge_id in region_topo_item['edges']]
        region_parsed = OspRegion(region_edge_list, region_topo_item['id'])

        region_dict[region_topo_item['id']] = region_parsed

    return region_dict


def parse_edge_end_points_by_id(point_id_list: list[str], vertices_topo_dict: dict):
    """
    将边的端点 point id 列表解析具体的 point 对象
    :param point_id_list:
    :param vertices_topo_dict:
    :return:
    """
    point_list = []
    if len(point_id_list) == 0:
        point_list = (None, None)

    else:

        for item in point_id_list:
            point_parsed = vertices_topo_dict[item]
            point_list.append(point_parsed)

    assert len(point_list) == 2
    return point_list


def parse_feat_topo(val2nd_ofs):
    topo = {}
    for val2nd_item_ofs in val2nd_ofs:
        val2nd_item_type = val2nd_item_ofs['message']['key']['message']['value']  # ['regions, 'edges', 'vertices']
        val4th_ofs = val2nd_item_ofs['message']['value']['message']['value']
        outer_list = []

        for val4th_item_ofs in val4th_ofs:
            # val4th_ofs: 每个元素代表 face、edge、vert 的一个元素
            val5th_ofs = val4th_item_ofs['message']['value']
            geo_dict = {}

            for val5th_item_ofs in val5th_ofs:
                # val5th_ofs: 每个元素代表 face、edge、vert 的一个具体属性，例如 id，坐标系等
                elem_type = val5th_item_ofs['message']['key']['message']['value']  # ['id', 'regions',  'edges', 'vertices']
                val6th_ofs = val5th_item_ofs['message']['value']

                if elem_type == 'param':
                    if val2nd_item_type == 'faces':
                        v = parse_region_msg(val6th_ofs)

                    elif val2nd_item_type == 'edges':
                        v = parse_edge_msg(val6th_ofs)

                    elif val2nd_item_type == 'vertices':
                        v = parse_vertex_msg(val6th_ofs)

                    else:
                        raise NotImplementedError(f'elem not supported: {val2nd_item_type}')

                elif elem_type in ('vertices', 'edges'):
                    v = parse_last_id(val6th_ofs['message']['value'])

                elif elem_type == 'id':
                    v = val6th_ofs['message']['value']

                elif elem_type == 'midpoint':
                    v = parse_last_msg_val_list(val6th_ofs['message']['value'])

                else:
                    warn(f'not considered key occurred: {elem_type}, parsed as [message][value]')
                    v = val6th_ofs['message']['value']

                geo_dict[elem_type] = v
            outer_list.append(geo_dict)

        topo[val2nd_item_type] = outer_list

    return topo


def parse_region_msg(val6th_ofs):
    """
    区域只保存 id、所在面定义、边界的 id
    """
    val7th_ofs = val6th_ofs['message']['value']
    # face_param = {'typeTag': val6th_ofs['message']['typeTag']}
    face_param = {}

    for val7th_item_ofs in val7th_ofs:
        k = val7th_item_ofs['message']['key']['message']['value']
        val9th_ofs = val7th_item_ofs['message']['value']['message']['value']

        if k == 'coordSystem':
            v = parse_coord_msg(val9th_ofs)

        elif k in ('normal', 'origin', 'x'):
            v = parse_last_msg_val_list(val9th_ofs)

        elif k in ('surfaceType', ):
            v = val9th_ofs

        else:
            warn(f'not considered key occurred: {k}, save directly')
            v = val9th_ofs

        face_param[k] = v

    return face_param


def parse_coord_msg(response):
    """
    parse coordSystem parameters from OnShape response data
    """
    coord_param = {}
    for item in response:
        k = item['message']['key']['message']['value']

        v_msg = item['message']['value']['message']['value']

        if k in ('origin', 'xAxis', 'zAxis'):
            v = parse_last_msg_val_list(v_msg)

        else:
            warn(f'not considered key occurred: {k}, parsed as origin')
            v = parse_last_msg_val_list(v_msg)

        coord_param[k] = v
    return coord_param


def parse_edge_msg(val6th_ofs):
    """
    edge 仅包含 id、edge 定义、edge 下的边的 id
    """
    assert isinstance(val6th_ofs, dict)

    val7th_ofs = val6th_ofs['message']['value']
    # edge_param = {'typeTag': val6th_ofs['message']['typeTag']}  # ['Circle', 'Line']
    edge_topo = {}

    for val7th_item_ofs in val7th_ofs:
        k = val7th_item_ofs['message']['key']['message']['value']  # 'coordSystem'
        val9th_item_ofs = val7th_item_ofs['message']['value']['message']['value']

        if k == 'curveType':
            v = val9th_item_ofs

        elif k in ('direction', 'origin', 'knots', 'weights'):
            v = parse_last_msg_val_list(val9th_item_ofs)

        elif k == 'coordSystem':
            v = parse_coord_msg(val9th_item_ofs)

        elif k in ('radius', 'degree', 'dimension', 'isPeriodic', 'isRational', 'majorRadius', 'minorRadius'):
            v = parse_last_msg_val(val7th_item_ofs['message']['value'])

        elif k == 'controlPoints':
            v = parse_past_last_msg_val_list(val9th_item_ofs)

        else:
            warn(f'not considered key occurred: {k}, save directly')
            v = val9th_item_ofs

        edge_topo[k] = v

    return edge_topo


def parse_last_msg_val_list(last_msg_val_list_ofs):
    """
    解析最深的仅需['message']['value']即可获取值的数组
    """
    assert isinstance(last_msg_val_list_ofs, list)

    val_parsed_list = []
    for item in last_msg_val_list_ofs:
        val_parsed = parse_last_msg_val(item)
        val_parsed_list.append(val_parsed)

    return val_parsed_list


def parse_past_last_msg_val_list(past_last_msg_val_list):
    """
    解析二重列表，最深一层可能是点坐标
    """
    assert isinstance(past_last_msg_val_list, list)
    val_parsed_list = []
    for item in past_last_msg_val_list:
        parsed_item = parse_last_msg_val_list(item['message']['value'])
        val_parsed_list.append(parsed_item)

    return val_parsed_list


def parse_last_msg_val(last_value_ofs):
    """
    解析最深的仅需['message']['value']即可获取值的字典
    """
    val_parsed = last_value_ofs['message']['value']

    # 如果该值有单位
    if 'unitToPower' in last_value_ofs['message']:
        unit = last_value_ofs['message']['unitToPower']
        assert len(unit) == 1
        unit = unit[0]

        # 获取单位
        mul_unit = utils.get_unit_trans_coff((unit['key'], unit['value']), macro.GLOBAL_UNIT)

        # 进行单位转换
        val_parsed *= mul_unit

    return val_parsed


def parse_last_id(last_msg_val_list_ofs):
    """
    解析最深的仅需['message']['value']即可获取 id 的数组
    """
    assert isinstance(last_msg_val_list_ofs, list)

    id_parsed_list = []
    for item in last_msg_val_list_ofs:
        id_parsed = item['message']['value']
        id_parsed_list.append(id_parsed)

    return id_parsed_list


def parse_vertex_msg(val6th_ofs):
    """
    parse vertex parameters from OnShape response data
    """
    assert isinstance(val6th_ofs, dict)
    vertices = parse_last_msg_val_list(val6th_ofs['message']['value'])

    return vertices


