"""
登录onshape的配置文件：
./config/onshape_credit.json

文件内容示例：
{
  "onshape_url": "https://cad.onshape.com",
  "access_key": "your_access_key",
  "secret_key": "your_secret_key"
}

ofs: 表示原始的从 onshape 上下载的文件内容的一部分
OSP: onshape sequence parser 简写

"""
import os
import json
from datetime import datetime
from functions.onshape.OnshapeClient import OnshapeClient
from functions.onshape.SketchParser import Sketch
from functions.onshape import SketchParser
import matplotlib.pyplot as plt
from functions.onshape import macro


def plot_3d_sketch(sample_list):
    # 创建图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    for sample in sample_list:
        ax.plot(sample[:, 0], sample[:, 1], sample[:, 2], label='3D Sketch', color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def request_sketch_plane(skh_parser_list, save_path, model_url):
    """
    批量获取草图平面信息
    :param skh_parser_list:
    :param save_path:
    :param model_url:
    :return:
    """
    v_list = model_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    onshape_client = OnshapeClient()

    all_req_id = []
    for i, item in enumerate(skh_parser_list):
        all_req_id.append(item.sketch_plane_id)

    res = onshape_client.get_face_by_id(did, wid, eid, all_req_id)

    with open(save_path, 'w') as f:
        json.dump(res.json(), f, ensure_ascii=False, indent=4)

    return res.json()


def request_sketch_topology(all_sketch_id, model_url):
    """
    批量获取草图中的全部区域信息
    :param all_sketch_id: 草图节点 ID 列表
    :param model_url:
    :return:
    """
    v_list = model_url.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    onshape_client = OnshapeClient()

    # 一次请求全部的 sketch topo
    res = onshape_client.eval_multi_sketch_topology(did, wid, eid, all_sketch_id)
    return res


def test_parse_sketch():

    ofs_path = os.path.join(macro.SAVE_ROOT, 'orig_ofs.json')
    with open(ofs_path, 'r') as f:
        ofs = json.load(f)

    all_sketch_id = []
    for i, fea_item_ofs in enumerate(ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']

        if feat_type == 'newSketch':
            all_sketch_id.append(fea_item_ofs['message']['featureId'])

    # 请求服务器一次获取所有草图区域拓扑信息
    # all_sketch_topo = request_sketch_topology(all_sketch_id, macro.URL)
    # timestamp = datetime.now().strftime('%H_%M_%S')
    # target_path = os.path.join(macro.SAVE_ROOT, f'all_sketch_topo_{timestamp}.json')
    # with open(target_path, 'w') as f:
    #     json.dump(all_sketch_topo, f, ensure_ascii=False, indent=4)
    # print('save all sketch topo succeed!')

    # 解析全部草图参数
    all_topo_file = os.path.join(macro.SAVE_ROOT, 'all_sketch_topo_20_16_58.json')
    with open(all_topo_file, 'r') as f:
        all_sketch_topo = json.load(f)

    val1st_ofs = all_sketch_topo['result']['message']['value']

    # 获取每个平面的拓扑
    all_sketch_parsed = []
    for val1st_item_ofs in val1st_ofs:
        sketch_parsed = Sketch(val1st_item_ofs)
        all_sketch_parsed.append(sketch_parsed)

    # 获取绘图元素：
    all_plots = []
    for sketch in all_sketch_parsed:
        for region in sketch.region_list:
            for prim in region.primitive_list:
                all_plots.append(prim.sample())

    # 绘图
    plot_3d_sketch(all_plots)
    return all_sketch_parsed


