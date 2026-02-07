from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt2d, TColgp_Array1OfPnt
from OCC.Core.Geom import Geom_BSplineSurface, Geom_BSplineCurve
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Core.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Display.SimpleGui import init_display

import math
from onshape import macro


def get_unit_trans_coff(unit_name, power, trans_to):
    """
    获取单位转换参数（乘数）
    :param unit_name: 类似 'METER'
    :param power: 次方，表示平方米还是立方米等。1, 2, 3
    :param trans_to: ['METER', 'in'] 米和英寸
    :return:
    """
    # 转换相同，不必转换
    if unit_name == trans_to[0] or unit_name == trans_to[1]:
        return 1.0

    elif unit_name == 'METER' and trans_to[0] == 'in':
        return  macro.METER_TO_IN ** power

    elif unit_name == 'in' and trans_to[0] == 'METER':
        return  macro.IN_TO_METER ** power

    elif unit_name == 'RADIAN' and trans_to[1] == 'DEGREE':
        return macro.RADIAN_TO_DEGREE ** power

    elif unit_name == 'DEGREE' and trans_to[1] == 'RADIAN':
        return macro.DEGREE_TO_RADIAN ** power

    else:
        raise NotImplementedError





def batch_upsert_by_id(arr, new_items, target_key='id'):
    """
    新数组加入到旧数组时，先判断新数组内 'id' 对应的值是否在原数组已包含
    如果已包含，则更新，否则直接加入到旧数组
    """
    # 1️⃣ 先把原数组变成 id → dict 的索引
    index = {item[target_key]: item for item in arr}

    # 2️⃣ 批量更新 / 插入（自动替换同 id）
    for item in new_items:
        index[item[target_key]] = item

    # 3️⃣ 如需要 list 结构再转回
    return list(index.values())


def no_closure(func):
    """
    标记该函数不允许捕获任何外层局部变量
    """
    free_vars = func.__code__.co_freevars
    if free_vars:
        raise RuntimeError(
            f'Function {func.__qualname__} is declared no-closure, '
            f'but captures outer variables: {free_vars}'
        )
    return func


def show_osp_face_list(osp_face_list):
    assert isinstance(osp_face_list, list)
    occt_face_list = []

    for face_item in osp_face_list:
        occt_face_list.append(face_item.bspline_face)

    show_occt_entity_list(occt_face_list)


def show_osp_face_dict(osp_face_dict):
    assert isinstance(osp_face_dict, dict)
    occt_face_list = []

    for _, face_item in osp_face_dict.items():
        occt_face_list.append(face_item.bspline_face)

    show_occt_entity_list(occt_face_list)


def show_occt_entity_list(entity_list):
    display, start_display, _, _ = init_display()
    display.DisplayShape(entity_list, update=True)
    start_display()


def compress_knots(knot_vec, tol=1e-6):
    """
    将节点向量转化为数值加重数，从而适应 OCCT 风格
    例如节点向量为:
        (0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1)

    需要转化为:
        数值：(0, 0.25, 0.5, 0.75, 1)，
        重数：(3, 1, 1, 1, 3)
    """
    knots, mults = [], []
    for k in knot_vec:
        if not knots or abs(k - knots[-1]) > tol:
            knots.append(k)
            mults.append(1)
        else:
            mults[-1] += 1

    knots = make_array1_real(knots)
    mults = make_array1_int(mults)
    return knots, mults


def make_array1_real(vals):
    """
    将实数数组转化为 OCCT 风格
    """
    arr = TColStd_Array1OfReal(1, len(vals))
    for i, v in enumerate(vals):
        arr.SetValue(i+1, float(v))
    return arr


def make_array1_int(vals):
    """
    将整形数组转化为 OCCT 风格
    """
    arr = TColStd_Array1OfInteger(1, len(vals))
    for i, v in enumerate(vals):
        arr.SetValue(i+1, round(v))
    return arr


def make_bspline_curve2d(curve2d_info):
    """
    利用从 json 中解析的信息，构造 OCCT 的 2d bspline curve
    从而形成 Face 的 pcurve
    """
    # ---------- parameters ----------
    ctrl = curve2d_info["controlPoints"]
    degree = round(curve2d_info['degree'])
    is_periodic = curve2d_info['isPeriodic']
    is_rational = curve2d_info['isRational']
    knot_raw = curve2d_info['knots']

    # ---------- poles ----------
    poles = TColgp_Array1OfPnt2d(1, len(ctrl))
    for i, (u, v) in enumerate(ctrl):
        poles.SetValue(i + 1, gp_Pnt2d(float(u), float(v)))

    # ---------- knots ----------
    knot, mult = compress_knots(knot_raw)

    # ---------- weights ----------
    weights_arr = None
    if is_rational:
        w = curve2d_info["weights"]
        weights_arr = make_array1_real(w)

    # ---------- build curve ----------
    if is_rational:
        pcurve = Geom2d_BSplineCurve(
            poles,
            weights_arr,
            knot,
            mult,
            degree,
            False
        )
    else:
        pcurve = Geom2d_BSplineCurve(
            poles,
            knot,
            mult,
            degree,
            False
        )

    # onshape 的 periodic bspline curve 实际上是首尾相连的 bspline，因此需要通过 occt 函数转化
    if is_periodic:
        pcurve.SetPeriodic()

    return pcurve


def make_bspline_surface(surf_info):
    """
    利用从json中获取的信息构建 OCCT 的 bspline surface
    """
    # ---------- parameters ----------
    ctrl = surf_info["controlPoints"]
    is_rational = surf_info['isRational']
    is_u_periodic = surf_info['isUPeriodic']
    is_v_periodic = surf_info['isVPeriodic']
    degree_u = round(surf_info['uDegree'])
    knot_raw_u = surf_info['uKnots']
    degree_v = round(surf_info['vDegree'])
    knot_raw_v = surf_info['vKnots']

    # ---------- poles ----------
    m, n = len(ctrl), len(ctrl[0])
    poles = TColgp_Array2OfPnt(1, m, 1, n)
    for i in range(m):
        for j in range(n):
            x, y, z = ctrl[i][j]
            poles.SetValue(i+1, j+1, gp_Pnt(x, y, z))

    # ---------- weights ----------
    weights_arr = None
    if is_rational:
        w = surf_info["weights"]
        weights_arr = TColStd_Array2OfReal(1, m, 1, n)
        for i in range(m):
            for j in range(n):
                weights_arr.SetValue(i+1, j+1, w[i][j])

    # ---------- knots ----------
    u_knot, u_mult = compress_knots(knot_raw_u)
    v_knot, v_mult = compress_knots(knot_raw_v)

    # ---------- make surface ----------
    if is_rational:
        surface = Geom_BSplineSurface(
            poles,
            weights_arr,
            u_knot,
            v_knot,
            u_mult,
            v_mult,
            degree_u,
            degree_v,
            False,
            False
        )
    else:
        surface = Geom_BSplineSurface(
            poles,
            u_knot,
            v_knot,
            u_mult,
            v_mult,
            degree_u,
            degree_v,
            False,
            False
        )

    if is_u_periodic:
        surface.SetUPeriodic()

    if is_v_periodic:
        surface.SetVPeriodic()

    return surface


def weld_bspline_loop(loop_bspline_list, tol):
    """
    修复由于近似而产生的参数域误差过大，导致 loop 不闭合的问题
    目前做法是直接焊接到中点
    对于多条 pcurve，他们在列表中应该是首尾相连的
    对于一条 pcurve，它应该是首尾相连的
    """
    @no_closure
    def _get_weld_point(point1, point2, alert_tol):
        """
        计算两个点的焊接点，目前是两点中点
        """
        p_dist = math.dist(point1, point2)

        if p_dist > alert_tol:
            raise ValueError(f'too large parameter tolerance: {p_dist}, please check!')

        else:
            weld_target = [(x + y) / 2 for x, y in zip(point1, point2)]
            return weld_target

    n_loop_curve = len(loop_bspline_list)
    # 没有元素的情况，直接返回空数组
    if n_loop_curve == 0:
        pass

    # loop 里只有一条 pcurve
    elif n_loop_curve == 1:
        ctrl = loop_bspline_list[0]['controlPoints']

        # 计算焊接点
        weld_point = _get_weld_point(ctrl[0], ctrl[-1], tol)

        # 将焊接点改写进原字典
        loop_bspline_list[0]['controlPoints'][0] = weld_point
        loop_bspline_list[0]['controlPoints'][-1] = weld_point

    # loop 里只有多条 pcurve
    else:
        for i in range(n_loop_curve):
            i_next = 0 if i == n_loop_curve - 1 else i + 1

            bspline_this = loop_bspline_list[i]
            bspline_next = loop_bspline_list[i_next]

            # 计算焊接点
            this_end_weld_next_start = _get_weld_point(bspline_this['controlPoints'][-1], bspline_next['controlPoints'][0], tol)

            # 将焊接点改写进原字典
            loop_bspline_list[i]['controlPoints'][-1] = this_end_weld_next_start
            loop_bspline_list[i_next]['controlPoints'][0] = this_end_weld_next_start

    return loop_bspline_list


def make_bspline_face(bspline_face_info):
    """
    利用从 onshape 解析到的数据在 occt 中重构 BSpline Face
    """
    # ===== 1. 构造 BSpline Surface =====
    surf_info = bspline_face_info["bSplineSurface"]
    surface = make_bspline_surface(surf_info)

    # ===== 2. 外边界 Wire =====
    if bspline_face_info["boundaryBSplineCurves"]:
        # 只有一个外环
        outer_loop = bspline_face_info["boundaryBSplineCurves"]

        # 修复 loop 的始末点，防止因误差导致重构失败
        outer_loop = weld_bspline_loop(outer_loop, macro.UV_TOL)

        wire_builder = BRepBuilderAPI_MakeWire()
        # 一个外环可能由多条 bspline curve 组成
        for pcurve_info in outer_loop:
            pcurve = make_bspline_curve2d(pcurve_info)
            pcurve_edge = BRepBuilderAPI_MakeEdge(pcurve, surface).Edge()
            wire_builder.Add(pcurve_edge)
        outer_wire = wire_builder.Wire()

    # ===== 3. 构造 Face =====
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire, True)

    else:
    # ===== 3. 构造 Face =====
        face_builder = BRepBuilderAPI_MakeFace(surface, True)

    # ===== 4. 内环 =====
    inner_loops = bspline_face_info['innerLoopBSplineCurves']
    # 可能有多个内环
    for single_inner_loop in inner_loops:
        # 修复 loop 的始末点，防止因误差导致重构失败
        single_inner_loop = weld_bspline_loop(single_inner_loop, macro.UV_TOL)

        wire_builder = BRepBuilderAPI_MakeWire()

        # 每个内环可能由多条 bspline curve 组成
        for pcurve_info in single_inner_loop:
            pcurve = make_bspline_curve2d(pcurve_info)
            pcurve_edge = BRepBuilderAPI_MakeEdge(pcurve, surface).Edge()
            wire_builder.Add(pcurve_edge)
        face_builder.Add(wire_builder.Wire())

    face = face_builder.Face()

    # ===== 5. 修复拓扑 =====
    fixer = ShapeFix_Face(face)
    fixer.Perform()

    return fixer.Face()


def example_molding():
    # ========== 1. 控制点 ==========
    degree = 3

    poles = TColgp_Array1OfPnt(1, 7)
    poles.SetValue(1, gp_Pnt(0, 0, 0))
    poles.SetValue(2, gp_Pnt(2, 1, 0))
    poles.SetValue(3, gp_Pnt(4, -1, 0))
    poles.SetValue(4, gp_Pnt(6, 0, 0))
    poles.SetValue(5, gp_Pnt(4, 1, 0))
    poles.SetValue(6, gp_Pnt(2, -1, 0))
    poles.SetValue(7, gp_Pnt(0, 0, 0))  # 形成周期闭合趋势

    # ========== 2. Knot 向量 ==========
    knots = TColStd_Array1OfReal(1, 5)
    knots.SetValue(1, 0.0)
    knots.SetValue(2, 1.0)
    knots.SetValue(3, 2.0)
    knots.SetValue(4, 3.0)
    knots.SetValue(5, 4.0)

    # ========== 3. Multiplicities ==========
    mults = TColStd_Array1OfInteger(1, 5)
    mults.SetValue(1, 1)
    mults.SetValue(2, 1)
    mults.SetValue(3, 1)
    mults.SetValue(4, 1)
    mults.SetValue(5, 1)

    # ========== 4. 构造 Periodic BSpline ==========
    curve = Geom_BSplineCurve(
        poles,
        knots,
        mults,
        degree,
        True  # periodic flag
    )

    # ========== 5. 生成 Edge ==========
    edge = BRepBuilderAPI_MakeEdge(curve).Edge()

    wire_builder = BRepBuilderAPI_MakeWire()
    wire_builder.Add(edge)
    wire = wire_builder.Wire()

    face_builder = BRepBuilderAPI_MakeFace(wire)
    face = face_builder.Face()

    # 拉伸方向
    vec = gp_Vec(0, 0, 10)

    prism = BRepPrimAPI_MakePrism(face, vec)
    solid = prism.Shape()

    display, start_display, _, _ = init_display()

    display.DisplayShape(solid, update=True)
    start_display()



