"""
上层应用
"""

import os
import shutil
from utils import utils
from functional import svg, image
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from pathlib import Path
import hashlib


translate_dict = {
    "轴承": "bearing",
    "销": "pin",
    "涡轮": "turbine",
    "螺栓": "bolt",
    "衬套": "bushing",
    "脚轮": "caster",
    "风扇": "fan",
    "法兰": "flange",
    "齿轮": "gear",
    "接头": "joint",
    "键": "key",
    "螺母": "nut",
    "堵头": "plug",
    "带轮": "pulley",
    "挡圈": "retaining_ring",
    "铆钉": "rivet",
    "螺钉": "screw",
    "间隔器": "spacer",
    "弹簧": "spring",
    "链轮": "sprocket",
    "螺柱": "stud",
    "阀门": "valve",
    "垫圈": "washer",
}


def svg_to_img_batched(source_dir, target_dir):
    """
    批量转化并保存数据
    :param source_dir:
    :param target_dir:
    :return:
    """
    if os.path.exists(target_dir):
        user = input(f"目标文件夹已存在文件，删除它？(y / n)").strip()
        if user.lower() == 'y':
            print('delete dir ' + target_dir)
            shutil.rmtree(target_dir)
        else:
            exit(0)
    os.makedirs(target_dir, exist_ok=True)

    # 获取全部类别
    classes_all = utils.get_subdirs(source_dir)

    for c_class in tqdm(classes_all):
        c_save_dir = os.path.join(target_dir, c_class)
        os.makedirs(c_save_dir)

        c_svg_all = utils.get_allfiles(os.path.join(source_dir, c_class), 'svg')
        for idx, c_svg in enumerate(c_svg_all):
            c_img = os.path.join(c_save_dir, f'{idx}.png')
            svg_proc.svg_to_image(c_svg, c_img)


def png_process_for_sketch_project(source_dir, target_dir):
    """
    为国重项目处理智丞得到的文件设计的函数
    :param source_dir:
    :param target_dir:
    :return:
    """
    if os.path.exists(target_dir):
        user = input(f"目标文件夹已存在文件，删除它？(y / n)").strip()
        if user.lower() == 'y':
            print('delete dir ' + target_dir)
            shutil.rmtree(target_dir)
        else:
            exit(0)
    os.makedirs(target_dir, exist_ok=True)

    # 获取全部类别
    classes_all = utils.get_subdirs(source_dir)

    for c_class in tqdm(classes_all):
        c_save_dir = os.path.join(target_dir, c_class)
        os.makedirs(c_save_dir)

        c_raw_png_all = utils.get_allfiles(os.path.join(source_dir, c_class), 'png')
        for idx, c_raw_png in enumerate(c_raw_png_all):
            if 'sketch' in c_raw_png:
                raw_name = os.path.basename(c_raw_png).replace('_sketch', '')
                c_img = os.path.join(c_save_dir, raw_name)
                # img_proc.crop_upper(c_raw_png, c_raw_png)
                shutil.copy(c_raw_png, c_img)
                img_proc.remove_png_white_pixel(c_img)


def remove_png_white_pixel_batched(dir_path, remove_pixel=(255, 255, 255), workers=4):
    # 获取全部图片路径
    pictures_all = utils.get_allfiles(dir_path, 'png', False)

    work_func = partial(img_proc.remove_png_white_pixel, remove_pixel=remove_pixel)

    with Pool(processes=workers) as pool:
        _ = list(
            tqdm(
                pool.imap(work_func, pictures_all),
                total=len(pictures_all),
                desc='processing png'
            )
        )

    # for c_pic in tqdm(pictures_all, total=len(pictures_all)):
    #     # 删除周围黑色像素
    #     # remove_round_black(c_pic, pix_width)
    #
    #     # 删除白色像素
    #     remove_png_white_pixel(c_pic)


def proj_sketchrnn_data_copy(folder1, folder2, folder3):
    # 路径请自行修改
    # folder1 = r"D:\folder1"  # PNG 文件所在目录
    # folder2 = r"D:\folder2"  # 可能包含子文件夹的 TXT 目录
    # folder3 = r"D:\folder3"  # 目标目录

    # 确保目标目录存在
    os.makedirs(folder3, exist_ok=True)

    # 获取第一个文件夹中所有 PNG 文件名（不带扩展名）
    png_files = [os.path.splitext(f)[0] for f in os.listdir(folder1) if f.lower().endswith('.png')]

    # 遍历第二个文件夹（递归）
    for root, dirs, files in os.walk(folder2):
        for file in files:
            if file.lower().endswith('.txt'):
                name_no_ext = os.path.splitext(file)[0]
                # 如果文件名与png文件匹配
                if name_no_ext in png_files:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(folder3, file)
                    shutil.copy2(src_path, dst_path)
                    print(f"已复制: {src_path} -> {dst_path}")


def proj_sketch_aug(source_dir, target_dir):
    all_subdirs = utils.get_subdirs(source_dir)

    for c_sub_dir in all_subdirs:
        c_sub_dir_full = os.path.join(source_dir, c_sub_dir)
        c_target_dir_full = os.path.join(target_dir, c_sub_dir)

        os.makedirs(c_target_dir_full, exist_ok=True)
        img_proc.augment_folder(c_sub_dir_full, c_target_dir_full, 1.5)


def proj_replace_comma_with_space(folder):
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".txt"):
                path = os.path.join(root, fname)
                print("Processing:", path)

                # 读取文件
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 替换逗号为空格
                content = content.replace(',', ' ')

                # 覆盖写回
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)


def proj_flip_last_value_in_folder(folder):
    """
    将 x,y,s 中的s标志位0变1，1变0
    :param folder:
    :return:
    """
    for root, dirs, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".txt"):
                path = os.path.join(root, fname)
                print("Processing:", path)

                new_lines = []

                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            new_lines.append(line)
                            continue

                        # 自动识别分隔符
                        if ',' in line:
                            parts = line.split(',')
                            sep = ','
                        else:
                            parts = line.split()
                            sep = ' '

                        # 最后一个字段必须是 s
                        s = parts[-1]

                        if s == '1':
                            parts[-1] = '0'
                        elif s == '0':
                            parts[-1] = '1'
                        else:
                            # 如果不是 0/1，不修改
                            pass

                        new_line = sep.join(parts)
                        new_lines.append(new_line)

                # 写回原文件
                with open(path, 'w', encoding='utf-8') as f:
                    for line in new_lines:
                        f.write(line + '\n')


def proj_flatten_dir_1(target_dir):
    """
    将target_dir 下的一级子文件夹分别进行展平
    :param target_dir:
    :return:
    """
    all_subdirs = utils.get_subdirs(target_dir)

    for c_dir in all_subdirs:
        c_sub_dor = os.path.join(target_dir, c_dir)
        utils.flatten_folder(c_sub_dor)


def proj_rename_subfolders(folder_path: str, translation_dict: dict=translate_dict, dry_run: bool = True):
    """
    将 folder_path 下的一级子文件夹按照 translation_dict 进行中英文重命名

    参数:
        folder_path: 目标文件夹路径
        translation_dict: 翻译字典
            例如: {"旧名称1": "新名称1", "旧名称2": "新名称2"}
            支持 中文→英文 或 英文→中文
        dry_run: True=仅预览不实际改名, False=真正执行重命名
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        print(f"错误：文件夹不存在或不是目录 → {folder_path}")
        return

    print(f"目标文件夹：{folder.resolve()}")
    print(f"干跑模式（不实际改名）：{'开启' if dry_run else '关闭'}")
    print("-" * 60)

    renamed_count = 0
    skipped_count = 0

    # 遍历一级子文件夹
    for item in folder.iterdir():
        if not item.is_dir():
            continue  # 跳过文件，只处理文件夹

        old_name = item.name
        new_name = translation_dict.get(old_name)

        if new_name and new_name != old_name:
            old_path = item
            new_path = item.with_name(new_name)

            if new_path.exists():
                print(f"⚠️  跳过（目标已存在）：{old_name} → {new_name}")
                skipped_count += 1
            else:
                print(f"✓ 重命名：{old_name} → {new_name}")
                if not dry_run:
                    try:
                        old_path.rename(new_path)
                    except Exception as e:
                        print(f"✗ 重命名失败：{old_name} → {new_name}，错误：{e}")
                        skipped_count += 1
                        continue
                renamed_count += 1
        else:
            if new_name is None:
                print(f"○ 无对应翻译，保持原名：{old_name}")
            else:
                print(f"○ 名称未变化：{old_name}")
            skipped_count += 1

    print("-" * 60)
    print(f"完成！实际重命名：{renamed_count} 个，跳过/无需改：{skipped_count} 个")
    if dry_run:
        print("这是预览模式，再次运行时将 dry_run=False 即可真正执行重命名")


def shorten_name_keep2(name, hash_len=8):
    """
    将 name 变成：前两个中文字符 + "_" + hash(剩余部分)

    示例：
    输入: "零件分类A_子系统_部件001"
    输出: "零件_3fa91c2b"
    """
    if len(name) <= 2:
        # 名字太短，不足以分割，直接 hash 整个名字
        h = hashlib.sha1(name.encode('utf-8')).hexdigest()[:hash_len]
        return name + "_" + h

    # 前两个字符
    prefix = name[:2]

    # 剩余部分
    rest = name[2:]

    # hash 剩余部分
    h = hashlib.sha1(rest.encode('utf-8')).hexdigest()[:hash_len]

    return f"{prefix}_{h}"


def proj_rename_deepest_dirs(folder_A):
    folder_A = os.path.abspath(folder_A)

    # 收集所有目录（越深的排在前）
    all_dirs = sorted(
        [d for d in os.walk(folder_A)],
        key=lambda x: x[0].count(os.sep),
        reverse=True
    )

    renamed = set()  # 避免重复处理

    for root, dirs, files in all_dirs:
        # 跳过根目录
        if root == folder_A:
            continue

        # 如果它包含子目录，说明不是最深层
        if dirs:
            continue

        # 避免一个目录被处理多次
        if root in renamed:
            continue

        # 相对路径，例如：x/y/z
        rel_path = os.path.relpath(root, folder_A)

        # 将路径变成 x_y_z
        new_name = rel_path.replace(os.sep, "_")
        new_name = shorten_name_keep2(new_name)
        new_path = os.path.join(folder_A, new_name)

        # 如果已经存在同名目录，需跳过或处理冲突
        if os.path.exists(new_path):
            print(f"Skipping (target exists): {new_path}")
            continue

        os.rename(root, new_path)
        renamed.add(new_path)
        print(f"Renamed: {root} → {new_path}")


def proj_check_deepest_dirs(folder):
    """
    统计最深的一层目录中的文件数，如果不为5，输出它。如果某个文件不在最深层目录，也输出他
    :param folder:
    :return:
    """
    folder = os.path.abspath(folder)

    # 保存结果
    wrong_dir_files = []   # 文件不在最深层目录
    wrong_count_dirs = []  # 深层目录文件数 ≠ 5

    # 第一步：找出所有最深层目录（即没有子目录的目录）
    deepest_dirs = set()

    for root, dirs, files in os.walk(folder):
        if not dirs:  # 没有子目录 → 最深层目录
            deepest_dirs.add(root)

    # 第二步：检查最深目录的文件数是否为 5
    for d in deepest_dirs:
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        if len(files) != 5:
            wrong_count_dirs.append((d, len(files)))

    # 第三步：检查所有文件是否都在最深层目录
    for root, dirs, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            # 文件所在目录不是最深层目录 → 输出
            if root not in deepest_dirs:
                wrong_dir_files.append(full_path)

    return wrong_count_dirs, wrong_dir_files


def proj_proj_rename_deepest_dirs2(base_dir = r'F:\sketch_proj\数据集\机械草图\5_真实采集手绘草图_76,935_(5_modal)'):
    """

    :param base_dir:
    :return:
    """
    all_subdirs = utils.get_subdirs(base_dir)
    for c_sud in all_subdirs:
        c_sub_full = os.path.join(base_dir, c_sud)

        proj_rename_deepest_dirs(c_sub_full)

    utils.remove_empty_dirs(base_dir)


def proj_classify_files(folder_A, categories):
    # 确保 folder_A 是绝对路径
    folder_A = os.path.abspath(folder_A)

    # 为每个类别创建文件夹
    for cat in categories:
        cat_dir = os.path.join(folder_A, cat)
        os.makedirs(cat_dir, exist_ok=True)

    # 遍历 A 下所有文件
    for filename in os.listdir(folder_A):
        file_path = os.path.join(folder_A, filename)

        # 跳过文件夹（刚创建的类别目录）
        if os.path.isdir(file_path):
            continue

        # 判断文件名中是否包含类别关键字
        for cat in categories:
            if cat.lower() in filename.lower():   # 不区分大小写
                dest_path = os.path.join(folder_A, cat, filename)
                shutil.move(file_path, dest_path)
                print(f"Moved: {filename} → {cat}/")
                break  # 避免一个文件被移动多次


if __name__ == '__main__':
    # png_process_for_sketch_project(r'E:\document\草图专项文件\项目结题\认知模型生成图\gen_jpg_svg', r'E:\document\草图专项文件\项目结题\认知模型生成图\sorted')
    # proj_sketchrnn_data_copy(r'C:\Users\ChengXi\Desktop\sketchrnn_data_proj', r'D:\document\DeepLearning\DataSet\sketch_cad\raw\sketch_txt_divide', r'C:\Users\ChengXi\Desktop\sketchrnn_proj_txt')
    # proj_sketch_aug(r'F:\deeplearning\草图项目\2_模型轮廓草图_1994591\MCB_A\train', r'F:\deeplearning\草图项目\轮廓扩充_mcb')

    # proj_replace_comma_with_space(r'E:\document\SolidEdge\backbone_2\sketchrnn_proj_txt')
    # proj_flip_last_value_in_folder(r'E:\document\SolidEdge\backbone_2\sketchrnn_proj_txt')

    # proj_flatten_dir_1(r'F:\deeplearning\机械草图\3_形变扩充草图_29531')
    # utils.remove_empty_dirs(r'F:\deeplearning\机械草图\3_形变扩充草图_29531')
    proj_rename_subfolders(r'F:\sketch_proj\数据集\机械草图_5,205,089\真实采集手绘草图_76,935_(5_modal)', dry_run=False)

    # proj_check_deepest_dirs(r'F:\sketch_proj\数据集\机械草图\5_真实采集手绘草图_76,935_(5_modal)')

    # proj_classify_files(r'F:\sketch_proj\数据集\机械草图\4_AI草图_25,590', list(translate_dict.keys()))

    # print(25590+1994591+86559+3021414+76935)

    pass











