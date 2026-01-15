import os
import glob
import yaml
import warnings
import cadquery as cq
from cadquery import exporters
from tqdm import tqdm
import time
import itertools

# ================= 系统配置 =================
# 输入：Raw Onshape YAML 数据集根目录
# INPUT_ROOT = r"/mnt/c/Users/grfpa/Downloads/12-9"
# 输出：生成的三维模型 (STEP) 存储目录
# OUTPUT_DIR = r"C:\Users\ChengXi\Desktop\cstnet2"
# 调试限制：0 或 -1 代表全量运行 (Production Mode)
# MAX_FILES = 0
# ===========================================


def parse_geometry_message(geo_msg):
    """
    几何解析器：从 YAML 消息中提取显式几何参数。
    应对策略：忽略隐式约束，只提取 Explicit Geometry。
    """
    scale = 1000.0  # 单位换算：米 -> 毫米
    data = {}

    # 1. 提取圆 (Circle) - 对应策略 3.1 基础图元还原
    if "radius" in geo_msg:
        data["type"] = "Circle"
        data["r"] = float(geo_msg["radius"]) * scale
        data["xc"] = float(geo_msg.get("xCenter", 0)) * scale
        data["yc"] = float(geo_msg.get("yCenter", 0)) * scale
        return data

    # 2. 提取线段 (Line Segment) - 兼容旧版 StartX/Y 格式
    if "startX" in geo_msg:
        data["type"] = "Line"
        data["x1"] = float(geo_msg["startX"]) * scale
        data["y1"] = float(geo_msg["startY"]) * scale
        data["x2"] = float(geo_msg["endX"]) * scale
        data["y2"] = float(geo_msg["endY"]) * scale
        return data

    # 3. 提取线段 (Line Segment) - 兼容新版 StartPoint 列表格式
    if "startPoint" in geo_msg and "endPoint" in geo_msg:
        try:
            sp, ep = geo_msg["startPoint"], geo_msg["endPoint"]
            data["type"] = "Line"
            data["x1"] = float(sp[0]) * scale
            data["y1"] = float(sp[1]) * scale
            data["x2"] = float(ep[0]) * scale
            data["y2"] = float(ep[1]) * scale
            return data
        except:
            pass

    # 4. 提取无限直线 (Infinite Line) - 对应策略 3.3：无限直线截断
    # 将点向式直线截断为 100mm 长的可视线段
    if "pntX" in geo_msg and "dirX" in geo_msg:
        try:
            px, py = float(geo_msg["pntX"]) * scale, float(geo_msg["pntY"]) * scale
            dx, dy = float(geo_msg["dirX"]), float(geo_msg["dirY"])
            half_len = 50.0
            data["type"] = "Line"
            data["x1"] = px - dx * half_len
            data["y1"] = py - dy * half_len
            data["x2"] = px + dx * half_len
            data["y2"] = py + dy * half_len
            return data
        except:
            pass

    # 5. [新增] 样条曲线 (Spline) - 对应策略 3.5：样条曲线支持
    # 解析 interpolationPoints 并拟合曲线
    if "interpolationPoints" in geo_msg:
        try:
            raw_pts = geo_msg["interpolationPoints"]
            # 数据解包：[x1, y1, x2, y2, ...] -> [(x,y), ...]
            pts = []
            for i in range(0, len(raw_pts), 2):
                x = float(raw_pts[i]) * scale
                y = float(raw_pts[i + 1]) * scale
                pts.append((x, y))

            if len(pts) >= 2:
                data["type"] = "Spline"
                data["points"] = pts
                data["is_closed"] = geo_msg.get("isPeriodic", False)
                return data
        except:
            pass

    return None


def process_single_file(file_path, save_path):
    """
    单文件处理流水线：YAML -> Sketch -> Extrude -> STEP
    """
    # 构造输出路径
    # folder_name = os.path.basename(os.path.dirname(file_path))
    # file_name = os.path.splitext(os.path.basename(file_path))[0]
    # save_path = os.path.join(OUTPUT_DIR, f"{folder_name}_{file_name}.step")

    # 断点续传检测
    if os.path.exists(save_path):
        return "Skipped"

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        solids_collection = []  # 实体收集容器
        features = data.get("features", [])

        # 遍历特征列表
        for feat in features:
            f_name = feat.get("message", {}).get("name", "Unknown")
            f_type = feat.get("typeName", "")

            # 仅处理 Sketch 相关特征 (过滤 Helix, Pattern 等复杂特征)
            if "Sketch" in f_type or "Sketch" in f_name or "Axis" in f_name:

                # 对应策略 3.2：坐标系扁平化 (强制使用 XY 平面)
                wp = cq.Workplane("XY")
                entities = feat.get("message", {}).get("entities", [])
                has_geo = False

                for ent in entities:
                    geo_msg = ent.get("message", {}).get("geometry", {}).get("message")
                    if not geo_msg:
                        continue
                    geo_data = parse_geometry_message(geo_msg)
                    if not geo_data:
                        continue

                    # CadQuery 绘图逻辑
                    try:
                        if geo_data["type"] == "Circle":
                            wp = wp.pushPoints(
                                [(geo_data["xc"], geo_data["yc"])]
                            ).circle(geo_data["r"])
                            has_geo = True

                        elif geo_data["type"] == "Line":
                            wp = wp.moveTo(geo_data["x1"], geo_data["y1"]).lineTo(
                                geo_data["x2"], geo_data["y2"]
                            )
                            has_geo = True

                        elif geo_data["type"] == "Spline":
                            pts = geo_data["points"]
                            try:
                                # 尝试光滑样条
                                wp = wp.moveTo(pts[0][0], pts[0][1]).spline(
                                    pts[1:], closed=geo_data["is_closed"]
                                )
                            except:
                                # 降级策略：多段线 (Polyline)
                                wp = wp.moveTo(pts[0][0], pts[0][1]).polyline(pts[1:])
                                if geo_data["is_closed"]:
                                    wp = wp.close()
                            has_geo = True
                    except:
                        pass

                # 对应策略 3.1：全量强制拉伸 (Force Blind Extrude)
                if has_geo:
                    try:
                        # 尝试生成实体 (Solid)
                        res = wp.extrude(5.0)
                        solids_collection.append(res)
                    except:
                        # 对应策略 3.4：兜底线框导出 (Wireframe Fallback)
                        try:
                            wires = wp.vals()
                            solids_collection.extend(wires)
                        except:
                            pass

        # 导出逻辑
        if solids_collection:
            assembly = cq.Assembly()
            for idx, obj in enumerate(solids_collection):
                assembly.add(obj, name=f"obj_{idx}")
            assembly.save(save_path, exportType="STEP")
            return "Success"
        else:
            return "Empty"  # 对应 Import 文件或纯约束草图

    except Exception as e:
        return f"Error: {str(e)}"


def main():
    import warnings

    warnings.filterwarnings("ignore")  # 屏蔽 FutureWarning

    # 路径自动修正 (兼容 WSL/Windows 路径格式)
    linux_input_root = INPUT_ROOT
    if ":" in INPUT_ROOT:
        linux_input_root = INPUT_ROOT.replace(":", "").replace("\\", "/")
        if not linux_input_root.startswith("/mnt/"):
            drive = linux_input_root[0].lower()
            path = linux_input_root[1:]
            linux_input_root = f"/mnt/{drive}{path}"

    print(f"📂 Input Path: {linux_input_root}")
    if not os.path.exists(linux_input_root):
        print("❌ Path not found!")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("🚀 Starting Stream Processing (Pipeline Ready)...")

    stats = {"Success": 0, "Empty": 0, "Error": 0, "Skipped": 0}
    count = 0
    EMPTY_LOG_FILE = "log_empty_files.txt"

    # 对应策略 4.1：流式扫描 (Stream Processing)
    # 使用 iglob 迭代器避免预加载卡死
    patterns = [
        os.path.join(linux_input_root, "**", "*.yml"),
        os.path.join(linux_input_root, "**", "*.yaml"),
    ]
    file_iterator = itertools.chain(
        glob.iglob(patterns[0], recursive=True), glob.iglob(patterns[1], recursive=True)
    )

    pbar = tqdm(file_iterator, desc="Processing")

    for f in pbar:
        if MAX_FILES > 0 and count >= MAX_FILES:
            break

        with open("current_processing_file.txt", "w") as trace:
            trace.write(f)

        res = process_single_file(f)

        if res == "Success":
            stats["Success"] += 1
        elif res == "Empty":
            stats["Empty"] += 1
            with open(EMPTY_LOG_FILE, "a") as log:
                log.write(f"{f}\n")
        elif res == "Skipped":
            stats["Skipped"] += 1
        else:
            stats["Error"] += 1

        count += 1

    print("\nProcessing Complete!")
    print(stats)


if __name__ == "__main__":
    # main()
    target_file = r'D:\document\DeepLearning\DataSet\ABC\abc_seq\abc_0000_ofs_v00\00000516\00000516_3c4e14158ece451f8d1c7318_featurescript_002.yml'
    res = process_single_file(target_file)


