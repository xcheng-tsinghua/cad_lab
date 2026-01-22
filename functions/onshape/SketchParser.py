"""
目前解析时，



"""

from functions.onshape.OspPoint import OspPoint
from functions.onshape.OspGeom import OspLine, OspCircle, SketchPlane
from functions.onshape import utils, macro


class Region(object):
    """
    一个草图可能存在多个区域，拉伸、旋转等都是选中区域进行操作
    """
    def __init__(self, region_id, primitive_list):
        self.region_id = region_id
        self.primitive_list = primitive_list


class Sketch(object):
    """
    A parser for OnShape sketch feature list
    """
    def __init__(self, fea_item_ofs):
        self.feat_id = fea_item_ofs['message']['featureId']
        self.feat_name = fea_item_ofs['message']['name']

        # 获取草图平面的 id
        self.sketch_plane_id = parse_sketch_plane_id(fea_item_ofs)

        # 先不获取，为了节省 request 数，最后一起获取
        self.sketch_plane = None

        # 先不获取，为了节省 request 数，最后一起获取
        # 获取草图的区域，原始的草图元素
        self.sketch_topology = None

        # 获取草图的区域，拉伸、旋转等都是用的区域，而非草图
        self.region_list = None

        # 获取草图的基元，原始的草图元素
        self.primitive_list = parse_sketch_primitives(fea_item_ofs)

    def load_sketch_plane(self, res_msg_val_ofs_item):
        self.sketch_plane = SketchPlane.from_ofs(res_msg_val_ofs_item)

        # 将草图中的元素都转换成 3d
        for i, c_prim in enumerate(self.primitive_list):
            self.primitive_list[i].map_to_3d(self.sketch_plane)

    def load_sketch_topo(self, val1st_item_ofs):
        """
        将草图拓扑转化为区域
        :param val1st_item_ofs:
        :return:
        """
        # 获取原始的 topology
        val2nd_ofs = val1st_item_ofs['message']['value']
        self.sketch_topology = parse_sketch_topo(val2nd_ofs)

        # 将原始的 topology 解析为区域
        region_topo_list = self.sketch_topology['faces']

        vertices_topo_dict = {}  # 将 list 转化为 key 为 id 的字典
        for vertices_topo_item in self.sketch_topology['vertices']:
            vertices_topo_dict[vertices_topo_item['id']] = vertices_topo_item

        edges_topo_dict = {}
        for edges_topo_item in self.sketch_topology['edges']:
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
                        vert_unit = vert_topo['param']['unit']

                        point_parsed = OspPoint(vert_coor[0], vert_coor[1], vert_coor[2], False)

                        mul_unit = utils.get_unit_trans_coff(vert_unit, macro.GLOBAL_UNIT)

                        point_parsed = mul_unit * point_parsed
                        edge_point_parsed.append(point_parsed)

                    edge_parsed = OspLine(edge_point_parsed[0], edge_point_parsed[1], False)

                elif edge_type == 'Circle':
                    # 是圆，解析圆心三维坐标、所在平面法线、半径

                    # 找到坐标系
                    coor_ofs = edge_topo_item['param']['coordSystem']

                    # 获取单位
                    origin_ofs = coor_ofs['origin']
                    coor_unit = origin_ofs[0][1]
                    mul_unit = utils.get_unit_trans_coff(coor_unit, macro.GLOBAL_UNIT)

                    # 获取坐标系的方向，圆应该是在 xoy 平面内，原点同时也是圆心
                    origin = OspPoint(origin_ofs[0][0], origin_ofs[1][0], origin_ofs[2][0], False)

                    # 统一单位
                    origin = mul_unit * origin

                    # coor_x_ofs = coor_ofs['xAxis']
                    # coor_x = OspPoint(coor_x_ofs[0][0], coor_x_ofs[1][0], coor_x_ofs[2][0], False)

                    coor_z_ofs = coor_ofs['zAxis']
                    coor_z = OspPoint(coor_z_ofs[0][0], coor_z_ofs[1][0], coor_z_ofs[2][0], False)

                    # circle_plane = SketchPlane(origin, coor_z, coor_x, coor_unit)

                    # 获取半径
                    radius = edge_topo_item['param']['radius']

                    # 统一单位
                    radius = radius * mul_unit

                    # 构造圆
                    edge_parsed = OspCircle(origin, radius, coor_z, False)

                edge_list.append(edge_parsed)

            region_parsed = Region(region_id, edge_list)
            region_list.append(region_parsed)

        self.region_list = region_list








def parse_sketch_plane_id(fea_item_ofs):
    """
    获取草图所在平面的id
    :param fea_item_ofs:
    :return:
    """
    param_list = fea_item_ofs['message']['parameters']

    sketch_plane_id = None
    for i, param_item in enumerate(param_list):
        param_msg = param_item['message']
        param_id = param_msg['parameterId']

        if param_id == 'sketchPlane' and 'queries' in param_msg:
            sketch_plane_id = param_msg['queries'][0]['message']['geometryIds'][0]
    return sketch_plane_id


def parse_line_param(entity_item_msg_ofs):
    """
    从原始的信息中获取直线
    :param entity_item_msg_ofs:
    :return:
    """
    start_param = entity_item_msg_ofs['startParam']
    end_param = entity_item_msg_ofs['endParam']

    _entity_item_msg_geom_msg_ofs = entity_item_msg_ofs['geometry']['message']
    loc_x = _entity_item_msg_geom_msg_ofs['pntX']
    loc_y = _entity_item_msg_geom_msg_ofs['pntY']
    dir_x = _entity_item_msg_geom_msg_ofs['dirX']
    dir_y = _entity_item_msg_geom_msg_ofs['dirY']

    loc_line = OspPoint(loc_x, loc_y, 0, True)
    dir_line = OspPoint(dir_x, dir_y, 0, True)

    start = loc_line + start_param * dir_line
    end = loc_line + end_param * dir_line

    return OspLine(start, end, True)


def parse_circle_param(entity_item_msg_ofs):
    """
    从原始的信息中获取圆
    :param entity_item_msg_ofs:
    :return:
    """
    entity_item_msg_geom_msg_ofs = entity_item_msg_ofs['geometry']['message']

    radius = entity_item_msg_geom_msg_ofs['radius']
    x_cen = entity_item_msg_geom_msg_ofs['xCenter']
    y_cen = entity_item_msg_geom_msg_ofs['yCenter']
    center = OspPoint(x_cen, y_cen, 0, True)

    return OspCircle(center, radius, None, True)


def parse_sketch_primitives(fea_item_ofs):
    """
    解析草图中的几何元素
    :return:
    """
    entity_parsed_list = []

    entity_ofs_list = fea_item_ofs['message']['entities']
    for entity_item_ofs in entity_ofs_list:

        entity_item_msg_ofs = entity_item_ofs['message']
        entity_type = entity_item_msg_ofs['geometry']['typeName']
        entity_parsed = None

        if entity_type == 'BTCurveGeometryLine':
            entity_parsed = parse_line_param(entity_item_msg_ofs)

        elif entity_type == 'BTCurveGeometryCircle':
            entity_parsed = parse_circle_param(entity_item_msg_ofs)

        entity_parsed_list.append(entity_parsed)

    return entity_parsed_list


def parse_multi_sketch_topo(topo_ofs):
    """
    解析全部的草图拓扑
    :param topo_ofs:
    :return:
    """
    res_msg_val_ofs = topo_ofs['result']['message']['value']

    all_sketch_topo = []
    for topo_item_ofs in res_msg_val_ofs:
        val2nd_ofs = topo_item_ofs['message']['value']

        sketch_topo_parsed = parse_sketch_topo(val2nd_ofs)
        all_sketch_topo.append(sketch_topo_parsed)

    return all_sketch_topo


def parse_sketch_topo(val2nd_ofs):
    topo = {}
    for val2nd_item_ofs in val2nd_ofs:
        val2nd_item_type = val2nd_item_ofs['message']['key']['message']['value']  # ['faces, 'edges']
        val4th_ofs = val2nd_item_ofs['message']['value']['message']['value']
        outer_list = []

        for val4th_item_ofs in val4th_ofs:
            val5th_ofs = val4th_item_ofs['message']['value']
            geo_dict = {}

            for val5th_item_ofs in val5th_ofs:
                elem_type = val5th_item_ofs['message']['key']['message']['value']  # ['id', edges/vertices]
                val6th_ofs = val5th_item_ofs['message']['value']

                if elem_type == 'param':
                    if val2nd_item_type == 'faces':
                        v = parse_face_msg(val6th_ofs)[0]

                    elif val2nd_item_type == 'edges':
                        v = parse_edge_msg(val6th_ofs)[0]

                    elif val2nd_item_type == 'vertices':
                        v = parse_vertex_msg(val6th_ofs)[0]

                    else:
                        raise NotImplementedError

                elif isinstance(val6th_ofs['message']['value'], list):
                    v = [a['message']['value'] for a in val6th_ofs['message']['value']]

                else:
                    v = val6th_ofs['message']['value']

                geo_dict[elem_type] = v
            outer_list.append(geo_dict)

        topo[val2nd_item_type] = outer_list

    return topo


def parse_face_msg(response):
    """
    parse face parameters from OnShape response data
    """
    # data = response.json()['result']['message']['value']
    data = [response] if not isinstance(response, list) else response
    faces = []

    for item in data:
        face_msg = item['message']['value']
        face_type = item['message']['typeTag']
        face_param = {'type': face_type}

        for msg in face_msg:
            k = msg['message']['key']['message']['value']
            v_item = msg['message']['value']['message']['value']
            if k == 'coordSystem':
                v = parse_coord_msg(v_item)

            elif isinstance(v_item, list):
                v = [round(x['message']['value'], 8) for x in v_item]

            else:
                if isinstance(v_item, float):
                    v = round(v_item, 8)

                else:
                    v = v_item

            face_param[k] = v
        faces.append(face_param)
    return faces


def parse_coord_msg(response):
    """parse coordSystem parameters from OnShape response data"""
    coord_param = {}
    for item in response:
        k = item['message']['key']['message']['value']

        v_msg = item['message']['value']['message']['value']

        v = []
        for final_item in v_msg:
            c_val = final_item['message']['value']

            unit = None
            if 'unitToPower' in final_item['message']:
                init_ofs = final_item['message']['unitToPower'][0]
                unit = (init_ofs['key'], init_ofs['value'])

            c_val_with_unit = (c_val, unit)
            v.append(c_val_with_unit)

        coord_param[k] = v
    return coord_param


def parse_edge_msg(response):
    """
    parse edge parameters from OnShape response data
    """
    data = [response] if not isinstance(response, list) else response

    edges = []
    for item in data:
        edge_msg = item['message']['value']
        edge_type = item['message']['typeTag']  # ['Circle', 'Line']
        edge_param = {'type': edge_type}

        for msg in edge_msg:
            k = msg['message']['key']['message']['value']  # 'coordSystem'
            v_item = msg['message']['value']['message']['value']

            if k == 'coordSystem':
                v = parse_coord_msg(v_item)

            elif isinstance(v_item, list):
                v = [round(x['message']['value'], 8) for x in v_item]

            else:
                if isinstance(v_item, float):
                    v = round(v_item, 8)
                else:
                    v = v_item

            edge_param.update({k: v})
        edges.append(edge_param)
    return edges


def parse_vertex_msg(response):
    """parse vertex parameters from OnShape response data"""
    data = [response] if not isinstance(response, list) else response
    vertices = []

    for item in data:
        xyz_msg = item['message']['value']
        xyz_type = item['message']['typeTag']
        p = []

        for msg in xyz_msg:
            p.append(round(msg['message']['value'], 8))

        unit = xyz_msg[0]['message']['unitToPower'][0]
        unit_exp = (unit['key'], unit['value'])
        vertices.append({xyz_type: tuple(p), 'unit': unit_exp})

    return vertices


