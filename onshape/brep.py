"""
构造或处理 BRep 数据
"""
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt2d
from OCC.Core.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.ShapeFix import ShapeFix_Face
from OCC.Display.SimpleGui import init_display

import math
from onshape import macro


def compress_knots(knot_vec, tol=1e-12):
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


def make_bspline_surface(surf_json):
    """
    利用从json中获取的信息构建 OCCT 的 bspline surface
    """
    ctrl = surf_json["controlPoints"]
    m, n = len(ctrl), len(ctrl[0])

    poles = TColgp_Array2OfPnt(1, m, 1, n)
    for i in range(m):
        for j in range(n):
            x, y, z = ctrl[i][j]
            poles.SetValue(i+1, j+1, gp_Pnt(x, y, z))

    weights_arr = None
    if surf_json["isRational"]:
        w = surf_json["weights"]
        weights_arr = TColStd_Array2OfReal(1, m, 1, n)
        for i in range(m):
            for j in range(n):
                weights_arr.SetValue(i+1, j+1, w[i][j])

    u_knot, u_mult = compress_knots(surf_json["uKnots"])
    v_knot, v_mult = compress_knots(surf_json["vKnots"])

    if surf_json["isRational"]:
        surface = Geom_BSplineSurface(
            poles,
            weights_arr,
            make_array1_real(u_knot),
            make_array1_real(v_knot),
            make_array1_int(u_mult),
            make_array1_int(v_mult),
            round(surf_json["uDegree"]),
            round(surf_json["vDegree"]),
            False,
            False
        )
    else:
        surface = Geom_BSplineSurface(
            poles,
            make_array1_real(u_knot),
            make_array1_real(v_knot),
            make_array1_int(u_mult),
            make_array1_int(v_mult),
            round(surf_json["uDegree"]),
            round(surf_json["vDegree"]),
            False,
            False
        )

    return surface


def make_bspline_curve2d_edge(curve_json, surface):
    """
    利用从 json 中解析的信息，构造 OCCT 的 2d bspline curve
    从而形成 Face 的 pcurve
    """
    # ---------- poles ----------
    ctrl = curve_json["controlPoints"]
    poles = TColgp_Array1OfPnt2d(1, len(ctrl))
    for i, (u, v) in enumerate(ctrl):
        poles.SetValue(i + 1, gp_Pnt2d(float(u), float(v)))

    # ---------- knots ----------
    knot, mult = compress_knots(curve_json["knots"])

    # ---------- weights ----------
    weights_arr = None
    is_rational = curve_json.get("isRational", False)
    if is_rational:
        w = curve_json["weights"]
        weights_arr = make_array1_real(w)

    # ---------- build curve ----------
    if is_rational:
        pcurve = Geom2d_BSplineCurve(
            poles,
            weights_arr,
            make_array1_real(knot),
            make_array1_int(mult),
            round(curve_json["degree"]),
            False # OCCT 中的 periodic 和 onshape 中的 periodic 意义不一致，在 OCCT 中统一使用 periodic=false
        )
    else:
        pcurve = Geom2d_BSplineCurve(
            poles,
            make_array1_real(knot),
            make_array1_int(mult),
            round(curve_json["degree"]),
            False # OCCT 中的 periodic 和 onshape 中的 periodic 意义不一致，在 OCCT 中统一使用 periodic=false
        )

    return BRepBuilderAPI_MakeEdge(pcurve, surface).Edge()


def get_weld_point(point1, point2, tol):
    """
    计算两个点的焊接点，目前是两点中点
    """
    p_dist = math.dist(point1, point2)

    if p_dist > tol:
        raise ValueError(f'too large parameter tolerance: {p_dist}, please check!')

    else:
        weld_point = [(x + y) / 2 for x, y in zip(point1, point2)]
        return weld_point


def weld_bspline_loop(loop_bspline_list, tol):
    """
    修复由于近似而产生的参数域误差过大，导致 loop 不闭合的问题
    目前做法是直接焊接到中点
    对于多条 pcurve，他们在列表中应该是首尾相连的
    对于一条 pcurve，它应该是首尾相连的
    """
    n_loop_curve = len(loop_bspline_list)
    # 没有元素的情况，直接返回空数组
    if n_loop_curve == 0:
        pass

    # loop 里只有一条 pcurve
    elif n_loop_curve == 1:
        ctrl = loop_bspline_list[0]['controlPoints']

        # 计算焊接点
        weld_point = get_weld_point(ctrl[0], ctrl[-1], tol)

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
            this_end_weld_next_start = get_weld_point(bspline_this['controlPoints'][-1], bspline_next['controlPoints'][0], tol)

            # 将焊接点改写进原字典
            loop_bspline_list[i]['controlPoints'][-1] = this_end_weld_next_start
            loop_bspline_list[i_next]['controlPoints'][0] = this_end_weld_next_start

    return loop_bspline_list


def make_bspline_face(bspline_face_json):
    """
    利用从 onshape 解析到的数据在 occt 中重构 BSpline Face
    """
    # ===== 1. 构造 BSpline Surface =====
    surf_json = bspline_face_json["bSplineSurface"]
    surface = make_bspline_surface(surf_json)

    # ===== 2. 外边界 Wire =====
    if bspline_face_json["boundaryBSplineCurves"]:
        # 只有一个外环
        outer_loop = bspline_face_json["boundaryBSplineCurves"]

        # 修复 loop 的始末点，防止因误差导致重构失败
        outer_loop = weld_bspline_loop(outer_loop, macro.UV_TOL)

        wire_builder = BRepBuilderAPI_MakeWire()
        # 一个外环可能由多条 bspline curve 组成
        for pcurve in outer_loop:
            wire_builder.Add(make_bspline_curve2d_edge(pcurve, surface))
        outer_wire = wire_builder.Wire()

    # ===== 3. 构造 Face =====
        face_builder = BRepBuilderAPI_MakeFace(surface, outer_wire, True)

    else:
    # ===== 3. 构造 Face =====
        face_builder = BRepBuilderAPI_MakeFace(surface, True)

    # ===== 4. 内环 =====
    inner_loops = bspline_face_json['innerLoopBSplineCurves']
    # 可能有多个内环
    for single_inner_loop in inner_loops:
        # 修复 loop 的始末点，防止因误差导致重构失败
        single_inner_loop = weld_bspline_loop(single_inner_loop, macro.UV_TOL)

        wire_builder = BRepBuilderAPI_MakeWire()

        # 每个内环可能由多条 bspline curve 组成
        for pcurve in single_inner_loop:
            wire_builder.Add(make_bspline_curve2d_edge(pcurve, surface))
        face_builder.Add(wire_builder.Wire())

    face = face_builder.Face()

    # ===== 5. 修复拓扑 =====
    fixer = ShapeFix_Face(face)
    fixer.Perform()

    return fixer.Face()


def display(entity_list):
    display, start_display, _, _ = init_display()
    display.DisplayShape(entity_list, update=True)
    start_display()


