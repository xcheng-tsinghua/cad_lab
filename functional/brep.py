"""
构造或处理 BRep 数据
"""
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Pln, gp_Circ, gp_Ax2
from OCC.Core.Geom import Geom_Plane
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeFace
from OCC.Core.BRepLib import breplib
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Display.SimpleGui import init_display
from OCC.Core.ElCLib import elclib


def test_construct():
    aax = gp_Ax2(gp_Pnt(1, 1, 1), gp_Dir(-1, -1, -1))
    acirc = gp_Circ(aax, 1)

    cir_start = elclib.Value(0, acirc)
    cir_end = elclib.Value(1, acirc)





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

    inner_cir = BRepBuilderAPI_MakeEdge(acirc, 1, 0).Edge()
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






