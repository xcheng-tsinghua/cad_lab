"""
目前解析时

"""

from functions.onshape.OspGeomBase import OspPoint, OspCoordSystem
from functions.onshape.OspGeomEdge import OspLine, OspCircle, OspEllipse, OspBSpline
from functions.onshape import utils, macro
from warnings import warn


class Region(object):
    """
    一个草图可能存在多个区域，拉伸、旋转等都是选中区域进行操作
    """
    def __init__(self, primitive_list: list, topo_id: str):
        self.id = topo_id
        self.primitive_list = primitive_list


class Sketch(object):
    """
    包含草图中全部区域及其几何元素
    """
    def __init__(self, val1st_item_ofs):
        # 原始解析出的拓扑结构
        self.sketch_topology = parse_sketch_topo(val1st_item_ofs['message']['value'])

        # 草图的区域：拉伸、旋转等都是用的区域，而非草图
        self.region_list = parse_sketch_region(self.sketch_topology)


def parse_sketch_region(sketch_topology):
    """
    将草图拓扑转化为区域
    :param sketch_topology:
    :return:
    """
    # 将 list 转化为 key 为 id 的字典，便于查询
    edges_topo_dict = {item['id']: item for item in sketch_topology['edges']}

    # 将 list 转化为 key 为 id 的字典，便于查询
    vertices_topo_dict = {item['id']: item['param'] for item in sketch_topology['vertices']}

    # 将原始的 topology 解析为区域
    region_list = []
    for region_topo_item in sketch_topology['faces']:

        region_edge_list = parse_edges_by_id(region_topo_item['edges'], edges_topo_dict, vertices_topo_dict)
        region_parsed = Region(region_edge_list, region_topo_item['id'])

        region_list.append(region_parsed)

    return region_list


def parse_edges_by_id(edge_id_list, edges_topo_dict, vertices_topo_dict):
    """
    将 edge id 列表解析具体的 edge 对象
    :param edge_id_list:
    :param edges_topo_dict:
    :param vertices_topo_dict:
    :return:
    """
    edge_list = []
    for edge_id_item in edge_id_list:
        edge_topo_item = edges_topo_dict[edge_id_item]
        edge_type = edge_topo_item['param']['curveType']

        # 将对应边解析为几何对象
        if edge_type == 'LINE':
            # 是直线，解析两个点
            edge_points_parsed = parse_edge_end_points_by_id(edge_topo_item['vertices'], vertices_topo_dict)
            edge_parsed = OspLine(edge_points_parsed[0], edge_points_parsed[1], edge_id_item)

        elif edge_type == 'CIRCLE':
            # 是圆，解析圆心三维坐标、所在平面法线、半径、始末点

            # 找到所在的局部坐标系
            coord_sys = OspCoordSystem.from_parsed_ofs(edge_topo_item['param']['coordSystem'])

            # 获取半径
            radius = edge_topo_item['param']['radius']

            # 获取端点
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vertices_topo_dict)

            # 构造圆
            edge_parsed = OspCircle(coord_sys, radius, vertices[0], vertices[1], edge_id_item)

        elif edge_type == 'ELLIPSE':
            # 找到所在的局部坐标系
            coord_sys = OspCoordSystem.from_parsed_ofs(edge_topo_item['param']['coordSystem'])

            # 获取半长轴 majorRadius
            major_radius = edge_topo_item['param']['majorRadius']

            # 获取短长轴 minorRadius
            minor_radius = edge_topo_item['param']['minorRadius']

            # 获取端点
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vertices_topo_dict)

            # 构造椭圆
            edge_parsed = OspEllipse(coord_sys, major_radius, minor_radius, vertices[0], vertices[1], edge_id_item)

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
            vertices = parse_edge_end_points_by_id(edge_topo_item['vertices'], vertices_topo_dict)

            # 构造自由曲线
            edge_parsed = OspBSpline(ctrl_points, degree, dimension, is_periodic, is_rational, knots, weights, vertices[0], vertices[1], edge_id_item)

        else:
            raise NotImplementedError(f'unsupported edge type: {edge_type}')

        edge_list.append(edge_parsed)

    return edge_list


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
            vert_coord = vertices_topo_dict[item]

            point_parsed = OspPoint.from_list(vert_coord)
            point_list.append(point_parsed)

    assert len(point_list) == 2
    return point_list


def parse_sketch_topo(val2nd_ofs):
    topo = {}
    for val2nd_item_ofs in val2nd_ofs:
        val2nd_item_type = val2nd_item_ofs['message']['key']['message']['value']  # ['faces, 'edges', 'vertices']
        val4th_ofs = val2nd_item_ofs['message']['value']['message']['value']
        outer_list = []

        for val4th_item_ofs in val4th_ofs:
            # val4th_ofs: 每个元素代表 face、edge、vert 的一个元素
            val5th_ofs = val4th_item_ofs['message']['value']
            geo_dict = {}

            for val5th_item_ofs in val5th_ofs:
                # val5th_ofs: 每个元素代表 face、edge、vert 的一个具体属性，例如 id，坐标系等
                elem_type = val5th_item_ofs['message']['key']['message']['value']  # ['id', edges/vertices]
                val6th_ofs = val5th_item_ofs['message']['value']

                if elem_type == 'param':
                    if val2nd_item_type == 'faces':
                        v = parse_face_msg(val6th_ofs)

                    elif val2nd_item_type == 'edges':
                        v = parse_edge_msg(val6th_ofs)

                    elif val2nd_item_type == 'vertices':
                        v = parse_vertex_msg(val6th_ofs)

                    else:
                        raise NotImplementedError

                elif elem_type in ('vertices', 'edges'):
                    v = parse_last_id(val6th_ofs['message']['value'])

                elif elem_type == 'id':
                    v = val6th_ofs['message']['value']

                else:
                    warn(f'not considered key occurred: {elem_type}, parsed as [message][value]')
                    v = val6th_ofs['message']['value']

                geo_dict[elem_type] = v
            outer_list.append(geo_dict)

        topo[val2nd_item_type] = outer_list

    return topo


def parse_face_msg(val6th_ofs):
    """
    parse face parameters from OnShape response data
    """
    face_msg = val6th_ofs['message']['value']
    face_param = {'typeTag': val6th_ofs['message']['typeTag']}

    for msg in face_msg:
        k = msg['message']['key']['message']['value']
        v_item = msg['message']['value']['message']['value']

        if k == 'coordSystem':
            v = parse_coord_msg(v_item)

        elif k in ('normal', 'origin', 'x'):
            v = parse_last_msg_val_list(v_item)

        elif k in ('surfaceType', ):
            v = v_item

        else:
            warn(f'not considered key occurred: {k}, save directly')
            v = v_item

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
    parse edge parameters from OnShape response data
    """
    assert isinstance(val6th_ofs, dict)

    edge_msg = val6th_ofs['message']['value']
    edge_param = {'typeTag': val6th_ofs['message']['typeTag']}  # ['Circle', 'Line']

    for msg in edge_msg:
        k = msg['message']['key']['message']['value']  # 'coordSystem'
        v_ofs = msg['message']['value']['message']['value']

        if k == 'curveType':
            v = v_ofs

        elif k in ('direction', 'origin', 'knots', 'weights'):
            v = parse_last_msg_val_list(v_ofs)

        elif k == 'coordSystem':
            v = parse_coord_msg(v_ofs)

        elif k in ('radius', 'degree', 'dimension', 'isPeriodic', 'isRational', 'majorRadius', 'minorRadius'):
            v = parse_last_msg_val(msg['message']['value'])

        elif k == 'controlPoints':
            v = parse_past_last_msg_val_list(v_ofs)

        else:
            warn(f'not considered key occurred: {k}, save directly')
            v = v_ofs

        edge_param[k] = v

    return edge_param


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


