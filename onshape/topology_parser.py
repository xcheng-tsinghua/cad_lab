"""
解析草图和各种建模命令获得的 区域、边、点 拓扑
"""
from onshape.OspGeomBase import OspPoint, OspCoordSystem
from onshape.OspGeomCurve import OspLine, OspCircle, OspEllipse, OspBSpline
from onshape.OspGeomEntity import OspFace, OspBody
from onshape import on_utils, macro
from colorama import Fore, Style


def parse_vert_dict(feature_topology):
    """
    将草图拓扑解析为 id 到具体点的映射
    :param feature_topology:
    :return:
    """
    vertices_topo_dict = {}
    vert_id_set = set()

    for item in feature_topology['vertices']:
        vert_id = item['id']

        vert_id_set.add(vert_id)
        vertices_topo_dict[vert_id] = OspPoint.from_list(item['param'])

    # vertices_topo_dict = {item['id']: OspPoint.from_list(item['param']) for item in feature_topology['vertices']}
    return vertices_topo_dict, vert_id_set


def parse_edge_dict(feature_topology, vert_dict):
    """
    将草图拓扑解析为 id 到具体边的映射
    :param feature_topology:
    :param vert_dict:
    :return:
    """
    vert_edge_dict = vert_dict
    edge_id_set = set()

    for edge_topo_item in feature_topology['edges']:
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
            midpoint = OspPoint.from_list(edge_topo_item['midPoint'])

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
            midpoint = OspPoint.from_list(edge_topo_item['midPoint'])

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
            print(Fore.RED + 'edge type OTHER occurred, treated as LINE' + Style.RESET_ALL)
            edge_points_parsed = parse_edge_end_points_by_id(edge_topo_item['vertices'], vert_dict)
            edge_parsed = OspLine(edge_points_parsed[0], edge_points_parsed[1], edge_id)

        else:
            raise NotImplementedError(f'unsupported edge type: {edge_type}')

        edge_id_set.add(edge_id)
        vert_edge_dict[edge_id] = edge_parsed

    return vert_edge_dict, edge_id_set


def parse_face_dict(feature_topology, vert_edge_dict):
    """
    将草图拓扑转化为区域
    :param feature_topology:
    :param vert_edge_dict:
    :return:
    """
    # 将原始的 topology 解析为区域
    vert_edge_face_dict = vert_edge_dict
    face_id_set = set()

    for face_topo_item in feature_topology['faces']:
        face_id = face_topo_item['id']

        face_edge_list = [vert_edge_dict[edge_id] for edge_id in face_topo_item['edges']]
        face_parsed = OspFace(face_topo_item['approximateBSplineSurface'], face_edge_list, face_topo_item['id'])

        vert_edge_face_dict[face_id] = face_parsed
        face_id_set.add(face_id)

    return vert_edge_face_dict, face_id_set


def parse_body_dict(feature_topology, vert_edge_face_dict):
    """
    获取实体信息
    """
    vert_edge_face_body_dict = vert_edge_face_dict
    body_id_set = set()

    for body_topo_item in feature_topology['bodies']:
        face_id_list = body_topo_item['faces']
        if face_id_list:
            body_id = body_topo_item['id']
            face_list = [vert_edge_face_dict[face_id] for face_id in face_id_list]

            body_parsed = OspBody(face_list, body_id)

            body_id_set.add(body_id)
            vert_edge_face_body_dict[body_id] = body_parsed

    return vert_edge_face_body_dict, body_id_set


def parse_topo_dict(topo_parsed_all):
    """
    将全部拓扑解析成 {id: topo} 的形式
    通常需要将某个回滚状态下，全部建模特征构建的实体全部获取后，再调用该函数
    """
    # 获取全部角点
    vert_dict, vert_id_set = parse_vert_dict(topo_parsed_all)

    # 获取全部边
    vert_edge_dict, edge_id_set = parse_edge_dict(topo_parsed_all, vert_dict)

    # 获取全部面。实测 Face 中包含 Region，因此这里不考虑 Region
    vert_edge_face_dict, face_id_set = parse_face_dict(topo_parsed_all, vert_edge_dict)

    # 获取全部实体
    vert_edge_face_body_dict, body_id_set = parse_body_dict(topo_parsed_all, vert_edge_face_dict)

    return vert_edge_face_body_dict, vert_id_set, edge_id_set, face_id_set, body_id_set


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
                elem_type = val5th_item_ofs['message']['key']['message']['value']  # ['id', 'faces',  'edges', 'vertices']
                val6th_ofs = val5th_item_ofs['message']['value']

                if elem_type == 'param':
                    if val2nd_item_type in ('faces', 'regions'):
                        value = parse_face_msg(val6th_ofs)

                    elif val2nd_item_type == 'edges':
                        value = parse_edge_msg(val6th_ofs)

                    elif val2nd_item_type == 'vertices':
                        value = parse_last_msg_val_list(val6th_ofs['message']['value'])

                    else:
                        raise NotImplementedError(f'elem not supported: {val2nd_item_type}')

                elif elem_type in ('vertices', 'edges', 'faces'):
                    value = parse_last_id(val6th_ofs['message']['value'])

                elif elem_type == 'id':
                    value = val6th_ofs['message']['value']

                elif elem_type == 'midPoint':
                    value = parse_last_msg_val_list(val6th_ofs['message']['value'])

                elif elem_type == 'approximateBSplineSurface':
                    value = parse_bspline_face(val6th_ofs['message']['value'])

                else:
                    print(Fore.RED + f'not considered key occurred: {elem_type}, parsed as [message][value]' + Style.RESET_ALL)
                    value = val6th_ofs['message']['value']

                geo_dict[elem_type] = value
            outer_list.append(geo_dict)

        topo[val2nd_item_type] = outer_list

    return topo


def parse_body_msg(val6th_ofs):
    """
    实体只保存 id、边界的 id
    """
    val7th_ofs = val6th_ofs['message']['value']
    # face_param = {'typeTag': val6th_ofs['message']['typeTag']}
    face_param = {}

    for val7th_item_ofs in val7th_ofs:
        elem_type = val7th_item_ofs['message']['key']['message']['value']
        val9th_ofs = val7th_item_ofs['message']['value']['message']['value']

        if elem_type == 'coordSystem':
            value = parse_coord_msg(val9th_ofs)

        elif elem_type in ('normal', 'origin', 'x'):
            value = parse_last_msg_val_list(val9th_ofs)

        elif elem_type in ('surfaceType', ):
            value = val9th_ofs

        else:
            print(Fore.RED + f'not considered body element type occurred: {elem_type}, save directly' + Style.RESET_ALL)
            value = val9th_ofs

        face_param[elem_type] = value

    return face_param


def parse_face_msg(val6th_ofs):
    """
    区域只保存 id、所在面定义、边界的 id
    """
    val7th_ofs = val6th_ofs['message']['value']
    # face_param = {'typeTag': val6th_ofs['message']['typeTag']}
    face_param = {}

    for val7th_item_ofs in val7th_ofs:
        elem_type = val7th_item_ofs['message']['key']['message']['value']
        val9th_ofs = val7th_item_ofs['message']['value']['message']['value']

        if elem_type == 'coordSystem':
            value = parse_coord_msg(val9th_ofs)

        elif elem_type in ('normal', 'origin', 'x', 'uKnots', 'vKnots'):
            value = parse_last_msg_val_list(val9th_ofs)

        elif elem_type in ('surfaceType', 'isRational', 'isUPeriodic', 'isVPeriodic', 'uDegree', 'vDegree'):
            value = val9th_ofs

        elif elem_type in ('radius', 'minorRadius', 'halfAngle'):  # 可能是 tour，因此没有 major radius
            value = parse_last_msg_val(val7th_item_ofs['message']['value'])

        elif elem_type == 'controlPoints':
            value = parse_past_last_msg_val_list(val9th_ofs)

        else:
            print(Fore.RED + f'not considered face element type occurred: {elem_type}, save directly' + Style.RESET_ALL)
            value = val9th_ofs

        face_param[elem_type] = value

    return face_param


def parse_edge_msg(val6th_ofs):
    """
    edge 仅包含 id、edge 定义、edge 下的边的 id
    """
    assert isinstance(val6th_ofs, dict)

    val7th_ofs = val6th_ofs['message']['value']
    # edge_param = {'typeTag': val6th_ofs['message']['typeTag']}  # ['Circle', 'Line']
    edge_topo = {}

    for val7th_item_ofs in val7th_ofs:
        elem_type = val7th_item_ofs['message']['key']['message']['value']  # 'coordSystem'
        val9th_item_ofs = val7th_item_ofs['message']['value']['message']['value']

        if elem_type == 'curveType':
            value = val9th_item_ofs

        elif elem_type in ('direction', 'origin', 'knots', 'weights'):
            value = parse_last_msg_val_list(val9th_item_ofs)

        elif elem_type == 'coordSystem':
            value = parse_coord_msg(val9th_item_ofs)

        elif elem_type in ('radius', 'degree', 'dimension', 'isPeriodic', 'isRational', 'majorRadius', 'minorRadius'):
            value = parse_last_msg_val(val7th_item_ofs['message']['value'])

        elif elem_type == 'controlPoints':
            value = parse_past_last_msg_val_list(val9th_item_ofs)

        else:
            print(Fore.RED + f'not considered edge element type occurred: {elem_type}, save directly' + Style.RESET_ALL)
            value = val9th_item_ofs

        edge_topo[elem_type] = value

    return edge_topo


def parse_coord_msg(response):
    """
    parse coordSystem parameters from OnShape response data
    """
    coord_param = {}
    for item in response:
        elem_type = item['message']['key']['message']['value']

        v_msg = item['message']['value']['message']['value']

        if elem_type in ('origin', 'xAxis', 'zAxis'):
            value = parse_last_msg_val_list(v_msg)

        else:
            print(Fore.RED + f'not considered coordinate system element type occurred: {elem_type}, parsed as origin' + Style.RESET_ALL)
            value = parse_last_msg_val_list(v_msg)

        coord_param[elem_type] = value
    return coord_param


def parse_last_msg_val_list(last_msg_val_list_ofs):
    """
    解析最深的仅需['message']['value']即可获取值的数组

    自身是一个数组
    每个 item 的 ['message']['value'] 是具体的数
    返回这些数值组成的数组
    """
    assert isinstance(last_msg_val_list_ofs, list)

    val_parsed_list = []
    for item in last_msg_val_list_ofs:
        val_parsed = parse_last_msg_val(item)
        val_parsed_list.append(val_parsed)

    return val_parsed_list


def parse_bspline_face(val7th_ofs):
    """
    从获取的 json 中解析详细 BSpline Surface 定义
    """
    parsed_bspline_face = {}
    for val7th_item_ofs in val7th_ofs:
        elem_type = val7th_item_ofs['message']['key']['message']['value']

        if elem_type == 'bSplineSurface':  # bspline surface 定义
            parsed_bspline_face[elem_type] = parse_bspline_face_bspline_surface(val7th_item_ofs)

        elif elem_type == 'boundaryBSplineCurves':  # bspline surface 的边界线的参数空间中的曲线
            parsed_bspline_face[elem_type] = parse_bspline_pcurve(val7th_item_ofs['message']['value']['message']['value'])

        elif elem_type == 'innerLoopBSplineCurves':
            val9th_ofs = val7th_item_ofs['message']['value']['message']['value']
            inner_loops = []

            # 可能有多个 inner loop
            for val9th_item_ofs in val9th_ofs:
                val10th_ofs = val9th_item_ofs['message']['value']

                # 每个 inner loop 可能由多个 bspline curve 组成
                single_inner_loop = []
                for val10th_item_ofs in val10th_ofs:
                    single_inner_loop.append(parse_single_bspline_pcurve(val10th_item_ofs))

                inner_loops.append(single_inner_loop)

            parsed_bspline_face[elem_type] = inner_loops

        else:
            raise NotImplementedError

    return parsed_bspline_face


def parse_bspline_face_bspline_surface(val7th_item_ofs):
    """
    获取 BSpline Face 上的 BSpline 曲面参数方程
    """
    val9th_ofs = val7th_item_ofs['message']['value']['message']['value']
    parsed_bspline_surface = {}

    for val9th_item_ofs in val9th_ofs:
        elem_type = val9th_item_ofs['message']['key']['message']['value']
        val11th_ofs = val9th_item_ofs['message']['value']['message']['value']

        if elem_type == 'controlPoints':  # 解析控制点矩阵
            # 控制点矩阵由 m 行 n 列控制点组成

            ctrl_point_rows = []
            for val11th_item_ofs in val11th_ofs:
                # 每次循环获取一行控制点（n 个）

                val12th_ofs = val11th_item_ofs['message']['value']
                ctrl_point_single_row = parse_past_last_msg_val_list(val12th_ofs)
                ctrl_point_rows.append(ctrl_point_single_row)

            parsed_bspline_surface['controlPoints'] = ctrl_point_rows

        elif elem_type in ('isRational', 'isUPeriodic', 'isVPeriodic', 'surfaceType', 'uDegree', 'vDegree'):
            parsed_bspline_surface[elem_type] = val11th_ofs

        elif elem_type in ('uKnots', 'vKnots'):
            parsed_bspline_surface[elem_type] = parse_last_msg_val_list(val11th_ofs)

        elif elem_type == 'weights':
            all_weights = []
            for val11th_item_ofs in val11th_ofs:
                parsed_weights = parse_last_msg_val_list(val11th_item_ofs['message']['value'])
                all_weights.append(parsed_weights)

            parsed_bspline_surface[elem_type] = all_weights

        else:
            raise NotImplementedError

    return parsed_bspline_surface


def parse_bspline_pcurve(val9th_ofs):
    """
    获取 BSpline Face 的边界 2d curve
    """
    parsed_bspline_pcurves = []

    assert isinstance(val9th_ofs, list)
    for val9th_item_ofs in val9th_ofs:
        parsed_bspline_pcurve = parse_single_bspline_pcurve(val9th_item_ofs)
        parsed_bspline_pcurves.append(parsed_bspline_pcurve)

    return parsed_bspline_pcurves


def parse_single_bspline_pcurve(val9th_item_ofs):
    """
    每个曲面边界可能有很多的边，这个函数解析其中一条边的 bspline pcurve 方程
    """
    val10th_ofs = val9th_item_ofs['message']['value']

    parsed_single_bspline_pcurve = {}
    for val10th_item_ofs in val10th_ofs:
        elem_type = val10th_item_ofs['message']['key']['message']['value']
        val12th_ofs = val10th_item_ofs['message']['value']['message']['value']

        if elem_type == 'controlPoints':
            parsed_single_bspline_pcurve['controlPoints'] = parse_past_last_msg_val_list(val12th_ofs)

        elif elem_type in ('curveType', 'degree', 'dimension', 'isPeriodic', 'isRational'):
            parsed_single_bspline_pcurve[elem_type] = val12th_ofs

            if elem_type == 'curveType':
                assert val12th_ofs == 'SPLINE'

        elif elem_type == 'knots':
            parsed_single_bspline_pcurve[elem_type] = parse_last_msg_val_list(val12th_ofs)

        else:
            raise NotImplementedError

    return parsed_single_bspline_pcurve


def parse_past_last_msg_val_list(past_last_msg_val_list):
    """
    解析二重列表，最深一层可能是点坐标

    自身是一个 list，表示一组数据
    每个 item 的 ['message']['value'] 也是一个 list，表示一个数据，比如一个点
    item['message']['value'][idx]['message']['value'] 有具体数值，表示点的具体坐标，例如 x、y、z
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
    assert isinstance(last_value_ofs, dict)
    val_parsed = last_value_ofs['message']['value']

    # 如果该值有单位
    if 'unitToPower' in last_value_ofs['message']:
        unit = last_value_ofs['message']['unitToPower']
        assert len(unit) == 1
        unit = unit[0]

        # 获取单位
        mul_unit = on_utils.get_unit_trans_coff(unit['key'], unit['value'], macro.GLOBAL_UNIT)

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



