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
import numpy as np
import copy
from collections import OrderedDict
import math
from functions.onshape.OnshapeClient import OnshapeClient
from functions.onshape.SketchParser import SketchParser
import matplotlib.pyplot as plt


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


def process_one(link, is_load_ofs):
    """
    link: model link on onShape
    is_load_ofs: 是否直接从文件读取序列，而不是通过 onshape_api request?
    """
    # create instance of the OnShape client; change key to test on another stack
    onshape_client = OnshapeClient()

    v_list = link.split("/")
    did, wid, eid = v_list[-5], v_list[-3], v_list[-1]

    # filter data that use operations other than sketch + extrude
    ofs_json_file = r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\ofs_orig.json'
    if is_load_ofs:
        with open(ofs_json_file, 'r') as f:
            ofs_data = json.load(f)

    else:
        ofs_data = onshape_client.get_features(did, wid, eid).json()
        try:
            with open(ofs_json_file, 'w') as f:
                json.dump(ofs_data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(e)
            exit(0)

    # 如果存在除了拉伸之外的命令，直接跳过
    # for item in ofs_data['features']:
    #     if item['message']['featureType'] not in ['newSketch', 'extrude']:
    #         print(f"methods except newSketch and extrude occurred: {item['message']['featureType']}, not support!")
    #         return 0

    parser = FeatureListParser(onshape_client, did, wid, eid, ofs_data)
    result = parser.parse()

    if len(result["sequence"]) < 2:
        return 0

    return result


# def process_batch(source_dir, target_dir):
#     """
#     处理一个文件夹下全部的序列文件
#     :param source_dir:
#     :param target_dir:
#     :return:
#     """
#     DWE_DIR = source_dir
#     DATA_ROOT = os.path.dirname(DWE_DIR)
#     filenames = sorted(os.listdir(DWE_DIR))
#     for name in filenames:
#         truck_id = name.split('.')[0].split('_')[-1]
#         print("Processing truck: {}".format(truck_id))
#
#         save_dir = os.path.join(DATA_ROOT, "processed/{}".format(truck_id))
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#
#         dwe_path = os.path.join(DWE_DIR, name)
#         with open(dwe_path, 'r') as fp:
#             dwe_data = yaml.safe_load(fp)
#
#         total_n = len(dwe_data)
#         count = Parallel(n_jobs=10, verbose=2)(delayed(process_one)(data_id, link, save_dir)
#                                                for data_id, link in dwe_data.items())
#         count = np.array(count)
#         print("valid: {}\ntotal:{}".format(np.sum(count > 0), total_n))
#         print("distribution:")
#         for n in np.unique(count):
#             print(n, np.sum(count == n))


def test_parse_sketch():
    ofs_path = r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\ofs_orig.json'
    with open(ofs_path, 'r') as f:
        ofs = json.load(f)

    all_new_sketches = []
    for i, fea_item_ofs in enumerate(ofs['features']):
        feat_type = fea_item_ofs['message']['featureType']

        if feat_type == 'newSketch':
            sketch_obj = SketchParser(fea_item_ofs)
            all_new_sketches.append(sketch_obj)

    # v_list = 'https://cad.onshape.com/documents/f8d3a3b2ddfbc6077f810cbc/w/50c3f52b580a97326eb89747/e/a824129468cfbb9a5a7f6bd0'.split("/")
    # did, wid, eid = v_list[-5], v_list[-3], v_list[-1]
    # onshape_client = OnshapeClient()

    # all_req_id = []
    # for i, item in enumerate(all_new_sketches):
    #     all_req_id.append(item.sketch_plane_id)
    #
    # res = onshape_client.get_face_by_id(did, wid, eid, all_req_id).json()

    oskh_plane_json_file = rf'E:\document\DeeplearningIdea\multi_cmd_seq_gen\sketch_plane_all.json'
    # with open(oskh_plane_json_file, 'w') as f:
    #     json.dump(res, f, ensure_ascii=False, indent=4)

    with open(oskh_plane_json_file, 'r') as f:
        all_planes = json.load(f)['result']['message']['value']

    # 获取每个平面的参数
    for sketch, plane in zip(all_new_sketches, all_planes):
        sketch.load_sketch_plane(plane)

    # 获取绘图元素：
    all_plots = []
    for sketch in all_new_sketches:
        for prim in sketch.primitive_list:
            all_plots.append(prim.sample())

    # 绘图
    plot_3d_sketch(all_plots)

    return all_new_sketches


def test():
    data_examples = {
        # '00000352': 'https://cad.onshape.com/documents/4185972a944744d8a7a0f2b4/w/d82d7eef8edf4342b7e49732/e/b6d6b562e8b64e7ea50d8325',
        # '00001272': 'https://cad.onshape.com/documents/b53ece83d8964b44bbf1f8ed/w/6b2f1aad3c43402c82009c85/e/91cb13b68f164c2eba845ce6',
        # '00001616': 'https://cad.onshape.com/documents/8c3b97c1382c43bab3eb1b48/w/43439c4e192347ecbf818421/e/63b575e3ac654545b571eee6',
        '00000351345632': 'https://cad.onshape.com/documents/f8d3a3b2ddfbc6077f810cbc/w/50c3f52b580a97326eb89747/e/a824129468cfbb9a5a7f6bd0',

    }

    for data_id, link in data_examples.items():
        process_one(link, True)
        print('trans finished')








