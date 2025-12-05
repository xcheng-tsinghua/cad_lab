"""
目前用于处理用svg表达的草图
"""

from svgpathtools import svg2paths
import numpy as np
from shapely.geometry import LineString, Point
import cv2
from noise import pnoise1
from svgpathtools import svg2paths, wsvg, Line, QuadraticBezier, CubicBezier, Arc
import random


def svg_read(svg_path, pen_down=1, pen_up=0, min_length=0.05, is_back_stk_list=False):
    """
    从 svg 文件读取草图

    svg 的 'id' 需要包含数字
    svg 的颜色（'stroke' 标签）是 ['#fff', '#ffff', '#fffff', '#ffffff', 'white'] 中的一个，表示擦除之前的笔划
    svg 的曲线，目前仅提取始末点

    :param svg_path:
    :param pen_down: ==1 下一个点属于本笔划
    :param pen_up: ==0 下一个点不属于本笔划
    :param min_length: 不考虑小于该长度的笔划
    :param is_back_stk_list: 是否直接返回笔划数组，数组内每个元素为 n*2 np.ndarray
    :return: [n, 3] (x, y, s) or [arr_1:(n*2), arr_2:(n*2), ..., arr_n:(n*2)]
    """

    def _is_eraser(_color_name):
        """
        判断是否是橡皮擦
        白色视为橡皮擦
        :param _color_name:
        :return:
        """
        if _color_name in ['#fff', '#ffff', '#fffff', '#ffffff', 'white']:
            return True
        else:
            return False

    def _is_path(_id):
        """
        id 不包含数字视作不是笔划
        :param _id:
        :return:
        """
        has_digit = any(ch.isdigit() for ch in _id)
        return has_digit

    # 获取每段的路径和属性
    paths, attributes = svg2paths(svg_path)

    strokes = []
    for path, attr in zip(paths, attributes):
        if len(path) == 0 or not _is_path(attr['id']):
            continue

        # 对于每个笔迹，提取线宽和是否是橡皮擦，橡皮擦是白色的笔划
        if 'stroke-width' in attr.keys():
            c_width = int(attr['stroke-width'])
        elif 'stroke-width'.upper() in attr.keys():
            c_width = int(attr['stroke-width'.upper()])
        else:
            raise ValueError('no key stroke-width')

        if 'stroke' in attr.keys():
            c_is_eraser = _is_eraser(attr['stroke'])
        elif 'stroke'.upper() in attr.keys():
            c_is_eraser = _is_eraser(attr['stroke'.upper()])
        else:
            raise ValueError('no key stroke')

        # 提取第一段的始末点
        c_start = path[0].start
        c_stk = [(c_start.real, c_start.imag)]

        # 后面的各段只记录终点
        for segment in path:
            c_end = segment.end
            c_stk.append((c_end.real, c_end.imag))
        c_stk = np.array(c_stk)

        if c_is_eraser:  # 如果是白色，进行擦除操作
            strokes = remove_covered_and_split(strokes, c_stk, c_width)
        else:  # 如果是黑色，加入到历史笔划
            strokes.append(c_stk)

    # 在每个点的末尾加上笔划状态
    stroke_list_np = []
    for c_stk in strokes:
        if len(c_stk) < 2 or stroke_length(c_stk) < min_length:
            continue

        if is_back_stk_list:
            stroke_list_np.append(c_stk)
            continue

        # 先构建一个全为 pen_down 的ndarray
        n = len(c_stk)
        ones_col = np.full((n, 1), pen_down, dtype=c_stk.dtype)

        # 将最后一个设置为 pen_up
        ones_col[-1, 0] = pen_up

        # 拼接到 (x, y) 上
        c_stk = np.hstack((c_stk, ones_col))
        stroke_list_np.append(c_stk)

    if is_back_stk_list:
        return stroke_list_np

    else:
        stroke_list_np = np.vstack(stroke_list_np)
        return stroke_list_np


def stroke_length(stroke):
    """
    笔划由 n*2 的 numpy 数组表达
    :param stroke:
    :return:
    """
    if stroke.shape[0] < 2:
        return 0.0

    stroke = stroke[:, :2]
    diffs = np.diff(stroke, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)

    return np.sum(segment_lengths)


def remove_covered_and_split(strokes, c_stk, c_width):
    """
    对 strokes 中的笔划进行擦除操作
    擦除路径为 c_stk，宽度 c_width
    如果某个笔划中间存在点被擦除，将在擦除点处分开为多个笔划

    :param strokes:
    :param c_stk:
    :param c_width:
    :return:
    """
    # 创建带宽度的笔划区域
    center_line = LineString(c_stk)
    stroke_area = center_line.buffer(c_width / 2, cap_style=2, join_style=2)

    # 最终新笔划列表
    new_strokes = []

    for stroke in strokes:
        sub_stroke = []  # 当前未被遮挡的子笔划

        for pt in stroke:
            point = Point(pt)
            if stroke_area.contains(point):
                # 如果当前点被遮挡：断开
                if len(sub_stroke) > 0:
                    new_strokes.append(np.array(sub_stroke))
                    sub_stroke = []
            else:
                sub_stroke.append(pt)

        # 收尾：最后一段未被遮挡的子笔划
        if len(sub_stroke) > 0:
            new_strokes.append(np.array(sub_stroke))

    return new_strokes


def svg_to_s3(svg_path, txt_path, pen_down=1, pen_up=0, delimiter=','):
    svg_data = svg_read(svg_path, pen_down, pen_up)
    np.savetxt(txt_path, svg_data, delimiter=delimiter)


def s3_to_img(sketch, image_size=(224, 224), line_thickness=1, pen_up=0, coor_mode='ABS', save_path=None):
    """
    将 S3 草图转化为 Tensor 图片
    sketch: np.ndarray

    x1, y1, s1
    x2, y2, s2
    ...
    xn, yn, sn

    x, y 为绝对坐标
    s = pen_down: 下一个点属于当前笔划
    s = pen_up: 下一个点不属于当前笔划
    注意 Quickdraw 中存储相对坐标，不能直接使用

    :param sketch: 文件路径或者加载好的 [n, 3] 草图
    :param image_size:
    :param line_thickness:
    :param pen_up:
    :return: list(image_size), 224, 224 为预训练的 vit 的图片大小
    """
    assert coor_mode in ['REL', 'ABS']
    width, height = image_size

    if isinstance(sketch, str):
        points_with_state = np.loadtxt(sketch)

    elif isinstance(sketch, np.ndarray):
        points_with_state = sketch

    else:
        raise TypeError('error sketch type')

    if coor_mode == 'REL':
        points_with_state[:, :2] = np.cumsum(points_with_state[:, :2], axis=0)

    # 1. 坐标归一化
    pts = np.array(points_with_state[:, :2], dtype=np.float32)
    states = np.array(points_with_state[:, 2], dtype=np.int32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    diff_xy = max_xy - min_xy

    if np.allclose(diff_xy, 0):
        scale_x = scale_y = 1.0
    else:
        scale_x = (width - 1) / diff_xy[0] if diff_xy[0] > 0 else 1.0
        scale_y = (height - 1) / diff_xy[1] if diff_xy[1] > 0 else 1.0
    scale = min(scale_x, scale_y)

    pts_scaled = (pts - min_xy) * scale
    pts_int = np.round(pts_scaled).astype(np.int32)

    offset_x = (width - (diff_xy[0] * scale)) / 2 if diff_xy[0] > 0 else 0
    offset_y = (height - (diff_xy[1] * scale)) / 2 if diff_xy[1] > 0 else 0
    pts_int[:, 0] += int(round(offset_x))
    pts_int[:, 1] += int(round(offset_y))

    # 2. 创建白色画布
    img = np.ones((height, width), dtype=np.uint8) * 255

    # 3. 笔划切分
    split_indices = np.where(states == pen_up)[0] + 1  # 下一个点是新笔划，所以+1
    strokes = np.split(pts_int, split_indices)

    # 4. 绘制每条笔划
    for stroke in strokes:
        if len(stroke) >= 2:  # 至少2个点才能画线
            stroke = stroke.reshape(-1, 1, 2)
            cv2.polylines(img, [stroke], isClosed=False, color=0, thickness=line_thickness, lineType=cv2.LINE_AA)

    # 5. 转为归一化float32 Tensor
    # tensor_img = torch.from_numpy(img).float() / 255.0

    if save_path is not None:
        cv2.imwrite(save_path, img)

    return img


def jitter_point(pt, amp, freq):
    t = random.random()
    dx = amp * pnoise1(t * freq)
    dy = amp * pnoise1((t + 10) * freq)
    return complex(pt.real + dx, pt.imag + dy)


def jitter_paths(paths, sigma=0.5):
    # sigma: 抖动强度
    new_paths = []
    for path in paths:
        new_segments = []
        for seg in path:
            # 所有控制点坐标都抖动一下
            seg = seg.__class__(
                *(p + (random.gauss(0, sigma) + random.gauss(0, sigma)*1j)
                  for p in seg)
            )
            new_segments.append(seg)
        new_paths.append(type(path)(*new_segments))
    return new_paths


def svg_to_image(svg_file, img_file):
    svg_s3_data = svg_read(svg_file)
    s3_to_img(svg_s3_data, line_thickness=2, save_path=img_file)


if __name__ == '__main__':
    c_svg = r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_svg_all\Bearing\00b11be6f26c85ca85f84daf52626b36_1.svg'
    save_svg = r'C:\Users\ChengXi\Desktop\svg_aug\1.svg'

    paths, attributes = svg2paths(c_svg)
    aug_svg = jitter_paths(paths)

    wsvg(aug_svg, attributes=attributes, filename=save_svg)


