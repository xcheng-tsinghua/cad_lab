"""
构造或处理 BRep 数据
"""
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Circ, gp_Ax2, gp_Elips, gp_Ax3, gp_Torus
from OCC.Core.Geom import Geom_Plane, Geom_Circle, Geom_ToroidalSurface
from OCC.Core.BRepLib import breplib
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Display.SimpleGui import init_display
from OCC.Core.ElCLib import elclib
import math
from OCC.Core.gp import gp_Pnt, gp_Pnt2d
from OCC.Core.TColgp import TColgp_Array2OfPnt, TColgp_Array1OfPnt2d
from OCC.Core.TColStd import TColStd_Array2OfReal, TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.Geom2d import Geom2d_BSplineCurve
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.ShapeFix import ShapeFix_Face


def construct_arc_edge(center: gp_Pnt, normal: gp_Dir, radius: float, start_point: gp_Pnt, end_point: gp_Pnt):
    """
    构建一个圆或者圆弧的 edge
    :param center: 三维空间中的圆心
    :param normal: 圆所在的法线方向，沿右手系旋转
    :param radius: 半径
    :param start_point: 三维起点，注意有效小数位要达到小数点后6位，整圆为 None
    :param end_point: 三维终点，注意有效小数位要达到小数点后6位，整圆为 None
    :return:
    """
    assert start_point is None and end_point is None or start_point is not None and end_point is not None
    circle_axis = gp_Ax2(center, normal)
    circle_3d = gp_Circ(circle_axis, radius)

    if start_point is None and end_point is None:
        circle_edge = BRepBuilderAPI_MakeEdge(circle_3d).Edge()
    else:
        circle_edge = BRepBuilderAPI_MakeEdge(circle_3d, start_point, end_point).Edge()

    return circle_edge


def construct_ellipse_edge(center: gp_Pnt, normal: gp_Dir, major_dir, major_radius: float, minor_radius: float, start_point: gp_Pnt, end_point: gp_Pnt):
    """
    构建一个圆或者圆弧的 edge
    :param center: 三维空间中的椭圆中心
    :param normal: 椭圆所在面的法线方向，沿右手系旋转
    :param major_dir: 椭圆半长轴方向，也是局部坐标系的 x 轴方向
    :param major_radius: 半长轴
    :param minor_radius: 半短轴
    :param start_point: 三维起点，注意有效小数位要达到小数点后6位，整圆为 None
    :param end_point: 三维终点，注意有效小数位要达到小数点后6位，整圆为 None
    :return:
    """
    assert start_point is None and end_point is None or start_point is not None and end_point is not None
    circle_axis = gp_Ax2(center, normal, major_dir)
    ellipse_3d = gp_Elips(circle_axis, major_radius, minor_radius)

    if start_point is None and end_point is None:
        circle_edge = BRepBuilderAPI_MakeEdge(ellipse_3d).Edge()
    else:
        circle_edge = BRepBuilderAPI_MakeEdge(ellipse_3d, start_point, end_point).Edge()

    return circle_edge


def test_construct_ellipse():
    cir_center = gp_Pnt(1, 1, 1)
    cir_normal = gp_Dir(1, 1, 1)
    elli_x = gp_Dir(1.5773503 - 1, 1.2113249 -1, 0.2113249 - 1)

    cir_start = gp_Pnt(1.5773503, 1.2113249, 0.2113249)
    cir_end = gp_Pnt(1.5163978, 0.7418011, 0.7418011)


    # =========================
    # 1️⃣ 构造几何平面
    # =========================
    origin = gp_Pnt(3, 0, 0)
    normal = gp_Dir(1, 1, 1)
    plane_geom = Geom_Plane(gp_Pln(origin, normal))

    # =========================
    # 2️⃣ 顶点
    # =========================
    A = gp_Pnt(3, 0, 0)
    B = gp_Pnt(0, 3, 0)
    C = gp_Pnt(0, 0, 3)

    a = gp_Pnt(2, 0.5, 0.5)
    b = gp_Pnt(0.5, 2, 0.5)
    c = gp_Pnt(0.5, 0.5, 2)

    # =========================
    # 3️⃣ 构造3D边
    # =========================
    AB = BRepBuilderAPI_MakeEdge(A, B).Edge()
    BC = BRepBuilderAPI_MakeEdge(B, C).Edge()
    CA = BRepBuilderAPI_MakeEdge(C, A).Edge()

    ab = BRepBuilderAPI_MakeEdge(a, b).Edge()
    bc = BRepBuilderAPI_MakeEdge(b, c).Edge()
    ca = BRepBuilderAPI_MakeEdge(c, a).Edge()

    inner_cir = construct_ellipse_edge(cir_center, cir_normal, elli_x, 1, 0.5, cir_start, cir_end)
    inner_lin = BRepBuilderAPI_MakeEdge(cir_start, cir_end).Edge()

    # =========================
    # 5️⃣ 外环
    # =========================
    outer_wire_mk = BRepBuilderAPI_MakeWire()
    outer_wire_mk.Add(AB)
    outer_wire_mk.Add(BC)
    outer_wire_mk.Add(CA)
    outer_wire = outer_wire_mk.Wire()

    # =========================
    # 6️⃣ 内环（孔，必须反向）
    # =========================

    inner_wire_mk = BRepBuilderAPI_MakeWire()
    # inner_wire_mk.Add(ab)
    # inner_wire_mk.Add(bc)
    # inner_wire_mk.Add(ca)

    inner_wire_mk.Add(inner_lin)
    inner_wire_mk.Add(inner_cir)
    inner_wire = inner_wire_mk.Wire()
    inner_wire.Reverse()

    # =========================
    # 7️⃣ 构造 Face
    # =========================
    face_mk = BRepBuilderAPI_MakeFace(plane_geom, outer_wire, True)
    face_mk.Add(inner_wire)
    face = face_mk.Face()

    # =========================
    # 4️⃣ 生成 pCurve（关键）
    # =========================
    # for e in [AB, BC, CA, ab, bc, ca]:
    #     breplib.BuildPCurveForEdgeOnPlane(e, face)

    # 设置容差
    builder = BRep_Builder()
    builder.UpdateFace(face, 1e-7)

    # 检查合法性
    print("Face valid:", BRepCheck_Analyzer(face).IsValid())

    # =========================
    # 8️⃣ 显示
    # =========================
    display, start_display, _, _ = init_display()
    display.DisplayShape(face, update=True)
    start_display()


def test_construct_circle():
    cir_center = gp_Pnt(1, 1, 1)
    cir_normal = gp_Dir(-1, -1, -1)

    cir_start = gp_Pnt(1.4082483, 1.4082483, 0.1835034)
    cir_end = gp_Pnt(0.1835034, 1.4082483, 1.4082483)


    # =========================
    # 1️⃣ 构造几何平面
    # =========================
    origin = gp_Pnt(3, 0, 0)
    normal = gp_Dir(1, 1, 1)
    plane_geom = Geom_Plane(gp_Pln(origin, normal))

    # =========================
    # 2️⃣ 顶点
    # =========================
    A = gp_Pnt(3, 0, 0)
    B = gp_Pnt(0, 3, 0)
    C = gp_Pnt(0, 0, 3)

    a = gp_Pnt(2, 0.5, 0.5)
    b = gp_Pnt(0.5, 2, 0.5)
    c = gp_Pnt(0.5, 0.5, 2)

    # =========================
    # 3️⃣ 构造3D边
    # =========================
    AB = BRepBuilderAPI_MakeEdge(A, B).Edge()
    BC = BRepBuilderAPI_MakeEdge(B, C).Edge()
    CA = BRepBuilderAPI_MakeEdge(C, A).Edge()

    ab = BRepBuilderAPI_MakeEdge(a, b).Edge()
    bc = BRepBuilderAPI_MakeEdge(b, c).Edge()
    ca = BRepBuilderAPI_MakeEdge(c, a).Edge()

    inner_cir = construct_arc_edge(cir_center, cir_normal, 1, cir_start, cir_end)
    inner_lin = BRepBuilderAPI_MakeEdge(cir_start, cir_end).Edge()

    # =========================
    # 5️⃣ 外环
    # =========================
    outer_wire_mk = BRepBuilderAPI_MakeWire()
    outer_wire_mk.Add(AB)
    outer_wire_mk.Add(BC)
    outer_wire_mk.Add(CA)
    outer_wire = outer_wire_mk.Wire()

    # =========================
    # 6️⃣ 内环（孔，必须反向）
    # =========================

    inner_wire_mk = BRepBuilderAPI_MakeWire()
    # inner_wire_mk.Add(ab)
    # inner_wire_mk.Add(bc)
    # inner_wire_mk.Add(ca)

    inner_wire_mk.Add(inner_lin)
    inner_wire_mk.Add(inner_cir)
    inner_wire = inner_wire_mk.Wire()
    inner_wire.Reverse()

    # =========================
    # 7️⃣ 构造 Face
    # =========================
    face_mk = BRepBuilderAPI_MakeFace(plane_geom, outer_wire, True)
    face_mk.Add(inner_wire)
    face = face_mk.Face()

    # =========================
    # 4️⃣ 生成 pCurve（关键）
    # =========================
    # for e in [AB, BC, CA, ab, bc, ca]:
    #     breplib.BuildPCurveForEdgeOnPlane(e, face)

    # 设置容差
    builder = BRep_Builder()
    builder.UpdateFace(face, 1e-7)

    # 检查合法性
    print("Face valid:", BRepCheck_Analyzer(face).IsValid())

    # =========================
    # 8️⃣ 显示
    # =========================
    display, start_display, _, _ = init_display()
    display.DisplayShape(face, update=True)
    start_display()


def construct_tour():
    atour = gp_Torus()



    # ========================
    # 1️⃣ 定义环面的几何坐标系
    # ========================
    origin = gp_Pnt(0, 0, 0)
    axis = gp_Dir(0, 0, 1)  # 环面绕 Z 轴
    xdir = gp_Dir(1, 0, 0)

    ax3 = gp_Ax3(origin, axis, xdir)

    R_major = 50  # 主半径（甜甜圈大圆）
    R_minor = 15  # 管半径（截面小圆）

    torus_surf = Geom_ToroidalSurface(ax3, R_major, R_minor)

    # ========================
    # 2️⃣ UV 参数裁剪区间
    # ========================
    u1 = 0
    u2 = 0.5 * math.pi  # 主圆方向范围

    v1 = 0
    v2 = math.pi  # 管截面半圈

    # ========================
    # 3️⃣ 构造 Face
    # ========================
    face = BRepBuilderAPI_MakeFace(torus_surf, u1, u2, v1, v2).Face()

    # ========================
    # 4️⃣ 显示
    # ========================
    display, start_display, _, _ = init_display()
    display.DisplayShape(face, update=True)
    display.FitAll()
    start_display()


# ---------- 工具函数 ----------
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

    u_knots, u_mult = compress_knots(surf_json["uKnots"])
    v_knots, v_mult = compress_knots(surf_json["vKnots"])

    if surf_json["isRational"]:
        surface = Geom_BSplineSurface(
            poles,
            weights_arr,
            make_array1_real(u_knots),
            make_array1_real(v_knots),
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
            make_array1_real(u_knots),
            make_array1_real(v_knots),
            make_array1_int(u_mult),
            make_array1_int(v_mult),
            round(surf_json["uDegree"]),
            round(surf_json["vDegree"]),
            False,
            False
        )

    return surface


def make_edge_from_curve2d(curve_json, surface):
    """
    利用从 json 中解析的信息，构造 OCCT 的 2d bspline curve
    从而形成 Face 的 pcurve
    """
    pts = curve_json["controlPoints"]
    degree = curve_json["degree"]
    # OCCT 中的 periodic 和 onshape 中的 periodic 意义不一致，在 OCCT 中统一使用 periodic=false
    # is_periodic = curve_json.get("isPeriodic", False)
    is_rational = curve_json.get("isRational", False)

    # ---------- poles ----------
    poles = TColgp_Array1OfPnt2d(1, len(pts))
    for i, (u, v) in enumerate(pts):
        poles.SetValue(i + 1, gp_Pnt2d(float(u), float(v)))

    # ---------- knots ----------
    knots, mults = compress_knots(curve_json["knots"])

    knots_arr = TColStd_Array1OfReal(1, len(knots))
    mults_arr = TColStd_Array1OfInteger(1, len(mults))
    for i, k in enumerate(knots):
        knots_arr.SetValue(i + 1, float(k))
        mults_arr.SetValue(i + 1, round(mults[i]))

    # ---------- weights ----------
    weights_arr = None
    if is_rational:
        w = curve_json["weights"]
        weights_arr = TColStd_Array1OfReal(1, len(w))
        for i, val in enumerate(w):
            weights_arr.SetValue(i + 1, float(val))

    # ---------- build curve ----------
    if is_rational:
        pcurve = Geom2d_BSplineCurve(
            poles,
            weights_arr,
            knots_arr,
            mults_arr,
            round(degree),
            False
        )
    else:
        pcurve = Geom2d_BSplineCurve(
            poles,
            knots_arr,
            mults_arr,
            round(degree),
            False
        )

    return BRepBuilderAPI_MakeEdge(pcurve, surface).Edge()


def fix_bspline_loop(loop_bspline_list, tol=1e-4):
    """
    修复由于近似而产生的参数域误差过大，导致 loop 不闭合的问题
    对于多条 pcurve，他们在列表中应该是首尾相连的
    对于一条 pcurve，它应该是首尾相连的
    """
    # 没有元素的情况，直接返回空数组
    if not loop_bspline_list:
        return []

    if len(loop_bspline_list == 1):
        ctrl = loop_bspline_list[0]['controlPoints']

        p_dist =

    for loop_bspline in loop_bspline_list:



def make_bspline_face(bspline_face_json):
    """
    利用从 onshape 解析到的数据在 occt 中重构 BSpline Face
    """
    # ===== 1. 构造 BSpline Surface =====
    surf_json = bspline_face_json["bSplineSurface"]
    surface = make_bspline_surface(surf_json)

    # ctrl = surf["controlPoints"]
    # m, n = len(ctrl), len(ctrl[0])
    #
    # poles = TColgp_Array2OfPnt(1, m, 1, n)
    # for i in range(m):
    #     for j in range(n):
    #         x, y, z = ctrl[i][j]
    #         poles.SetValue(i+1, j+1, gp_Pnt(x, y, z))
    #
    # weights_arr = None
    # if surf["isRational"]:
    #     w = surf["weights"]
    #     weights_arr = TColStd_Array2OfReal(1, m, 1, n)
    #     for i in range(m):
    #         for j in range(n):
    #             weights_arr.SetValue(i+1, j+1, w[i][j])
    #
    # u_knots, u_mult = compress_knots(surf["uKnots"])
    # v_knots, v_mult = compress_knots(surf["vKnots"])
    #
    # if surf["isRational"]:
    #     surface = Geom_BSplineSurface(
    #         poles,
    #         weights_arr,
    #         make_array1_real(u_knots),
    #         make_array1_real(v_knots),
    #         make_array1_int(u_mult),
    #         make_array1_int(v_mult),
    #         round(surf["uDegree"]),
    #         round(surf["vDegree"]),
    #         False,
    #         False
    #     )
    # else:
    #     surface = Geom_BSplineSurface(
    #         poles,
    #         make_array1_real(u_knots),
    #         make_array1_real(v_knots),
    #         make_array1_int(u_mult),
    #         make_array1_int(v_mult),
    #         round(surf["uDegree"]),
    #         round(surf["vDegree"]),
    #         False,
    #         False
    #     )

    # ===== 2. 外边界 Wire =====
    if bspline_face_json["boundaryBSplineCurves"]:
        # 只有一个外环
        outer_loop = bspline_face_json["boundaryBSplineCurves"]

        wire_builder = BRepBuilderAPI_MakeWire()
        # 一个外环可能由多条 bspline curve 组成
        for pcurve in outer_loop:
            wire_builder.Add(make_edge_from_curve2d(pcurve, surface))
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
        wire_builder = BRepBuilderAPI_MakeWire()

        # 每个内环可能由多条 bspline curve 组成
        for pcurve in single_inner_loop:
            wire_builder.Add(make_edge_from_curve2d(pcurve, surface))
        face_builder.Add(wire_builder.Wire())

    face = face_builder.Face()

    # ===== 5. 修复拓扑 =====
    fixer = ShapeFix_Face(face)
    fixer.Perform()

    return fixer.Face()


