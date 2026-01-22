"""
目前解析时，



"""

from functions.onshape.OspPoint import OspPoint
from functions.onshape.OspGeom import OspLine, OspCircle
from functions.onshape import utils, macro
from warnings import warn


class Region(object):
    """
    一个草图可能存在多个区域，拉伸、旋转等都是选中区域进行操作
    """
    def __init__(self, region_id, primitive_list):
        self.region_id = region_id
        self.primitive_list = primitive_list


class Sketch(object):
    """
    包含草图中全部区域及其几何元素
    """
    def __init__(self, val1st_item_ofs):
        # 草图的区域：拉伸、旋转等都是用的区域，而非草图
        self.sketch_topology, self.region_list = parse_sketch_topo_and_region(val1st_item_ofs)

def parse_sketch_topo_and_region(val1st_item_ofs):
    """
    将草图拓扑转化为区域
    :param val1st_item_ofs:
    :return:
    """
    # 获取原始的 topology
    val2nd_ofs = val1st_item_ofs['message']['value']
    sketch_topology = parse_sketch_topo(val2nd_ofs)

    # 将原始的 topology 解析为区域
    region_topo_list = sketch_topology['faces']

    vertices_topo_dict = {}  # 将 list 转化为 key 为 id 的字典
    for vertices_topo_item in sketch_topology['vertices']:
        vertices_topo_dict[vertices_topo_item['id']] = vertices_topo_item

    edges_topo_dict = {}
    for edges_topo_item in sketch_topology['edges']:
        edges_topo_dict[edges_topo_item['id']] = edges_topo_item

    region_list = []
    for region_topo_item in region_topo_list:
        edge_id_list = region_topo_item['edges']
        region_id = region_topo_item['id']

        edge_list = []
        for edge_id_item in edge_id_list:

            edge_topo_item = edges_topo_dict[edge_id_item]
            edge_type = edge_topo_item['param']['type']

            # 将对应边解析为几何对象
            if edge_type == 'Line':
                # 是直线，解析两个点
                edge_vert_id_list = edge_topo_item['vertices']
                assert len(edge_vert_id_list) == 2

                edge_point_parsed = []
                for edge_vert_id_item in edge_vert_id_list:
                    vert_topo = vertices_topo_dict[edge_vert_id_item]

                    vert_coor = vert_topo['param']['Vector']

                    point_parsed = OspPoint(vert_coor[0], vert_coor[1], vert_coor[2])

                    edge_point_parsed.append(point_parsed)

                edge_parsed = OspLine(edge_point_parsed[0], edge_point_parsed[1])

            elif edge_type == 'Circle':
                # 是圆，解析圆心三维坐标、所在平面法线、半径

                # 找到坐标系
                coor_ofs = edge_topo_item['param']['coordSystem']

                # 获取单位
                origin_ofs = coor_ofs['origin']

                # 获取坐标系的方向，圆应该是在 xoy 平面内，原点同时也是圆心
                origin = OspPoint(origin_ofs[0][0], origin_ofs[1][0], origin_ofs[2][0])

                coor_z_ofs = coor_ofs['zAxis']
                coor_z = OspPoint(coor_z_ofs[0][0], coor_z_ofs[1][0], coor_z_ofs[2][0])

                # 获取半径
                radius = edge_topo_item['param']['radius']

                # 构造圆
                edge_parsed = OspCircle(origin, radius, coor_z)

            edge_list.append(edge_parsed)

        region_parsed = Region(region_id, edge_list)
        region_list.append(region_parsed)

    return sketch_topology, region_list


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





        elif isinstance(v_ofs, list):
            v = [round(x['message']['value'], 8) for x in v_ofs]

        else:
            if isinstance(v_ofs, float):
                v = round(v_ofs, 8)
            else:
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


def parse_vertex_msg(val6th_ofs):
    """
    parse vertex parameters from OnShape response data
    """
    assert isinstance(val6th_ofs, dict)
    vertices = parse_last_msg_val_list(val6th_ofs['message']['value'])

    return vertices


