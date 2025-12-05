# pack
from PIL import Image
import cv2
import os
import random
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage import transform
from tqdm import tqdm


def remove_png_white_pixel(png_file, remove_pixel=(255, 255, 255)):
    # 打开图片
    img = Image.open(png_file).convert("RGBA")

    # 获取像素数据
    pixels = img.getdata()

    # 创建一个新的像素列表
    new_pixels = []
    for pixel in pixels:
        # 检查像素是否为白色
        if pixel[:3] == remove_pixel:
            # 如果是白色，则将其设置为透明
            new_pixels.append((255, 255, 255, 0))
        else:
            # 否则保持原样
            new_pixels.append(pixel)

    # 更新图像数据
    img.putdata(new_pixels)

    # 自动裁剪图片四周的空白区域（透明区域）
    cropped_img = img.crop(img.getbbox())

    # 保存修改后的图像
    cropped_img.save(png_file, "PNG")


def crop_upper(img_file, save_file, cut_pixels=40):
    """
    删除指定图片的上侧指定像素(png)
    :param img_file:
    :param save_file: 保存路径
    :param cut_pixels: 删除像素数
    :return:
    """
    # 读取图片
    img = Image.open(img_file)

    # 获取原始尺寸
    width, height = img.size

    # 裁剪：保留从 (0, cut_pixels) 到 (width, height)
    cropped = img.crop((0, cut_pixels, width, height))

    # 保存结果
    cropped.save(save_file)


# ------------------------------
# 各种变形函数
# ------------------------------

def affine_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    margin = 20
    pts2 = pts1 + np.float32([
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)]
    ])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h), borderValue=255, borderMode=cv2.BORDER_REPLICATE)


def perspective_transform(img):
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    margin = 40
    pts2 = pts1 + np.float32([
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)],
        [random.uniform(-margin, margin), random.uniform(-margin, margin)]
    ])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (w, h), borderValue=255, borderMode=cv2.BORDER_REPLICATE)


def elastic_transform(img, alpha=40, sigma=6):
    random_state = np.random.RandomState(None)
    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy, x + dx)
    distorted = map_coordinates(img, indices, order=1, mode='reflect')
    return distorted.astype(np.uint8)


def piecewise_affine_transform(img):
    rows, cols = img.shape[:2]
    src_cols = np.linspace(0, cols, 4)
    src_rows = np.linspace(0, rows, 4)
    src = np.dstack(np.meshgrid(src_cols, src_rows)).reshape(-1, 2)
    dst = src + np.random.normal(0, 15, src.shape)
    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src, dst)
    out = transform.warp(img, tform, output_shape=(rows, cols), mode='constant', cval=1.0)
    out = (out * 255).astype(np.uint8)
    return out


# ------------------------------------------
# 增强 pipeline（每次随机选一个增强方法）
# ------------------------------------------
AUG_FUNCS = [
    affine_transform,
    perspective_transform,
    elastic_transform,
    piecewise_affine_transform,
]


def random_augment(img):
    func = random.choice(AUG_FUNCS)
    return func(img)


# ------------------------------------------
# 主程序：扩增 1.5 倍
# ------------------------------------------

def augment_folder(input_dir, output_dir, ratio=1.5):
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    N = len(files)
    extra = int(N * ratio)  # 例如 ratio=1.5 → 额外生成 N*0.5 张

    print(f"原始数据 {N} 张，需要扩增 {ratio} 倍 → 新增 {extra} 张")

    # 直接复制原图
    # for f in files:
    #     # img = cv2.imread(os.path.join(input_dir, f), cv2.IMREAD_GRAYSCALE)
    #     img = np.array(Image.open(os.path.join(input_dir, f)).convert("L"))
    #     cv2.imwrite(os.path.join(output_dir, f), img)

    # 增强生成 extra 张图
    for i in tqdm(range(extra)):
        fname = files[i % N]
        # img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
        source_file = os.path.join(input_dir, fname)
        img = np.array(Image.open(source_file).convert("L"))

        aug = random_augment(img)

        outname = f"{os.path.splitext(fname)[0]}_aug{i}.png"
        outname = os.path.join(output_dir, outname)
        # cv2.imwrite(outname, aug)
        Image.fromarray(aug).save(outname)

    print("完成！")


if __name__ == '__main__':
    # crop_upper(r'C:\Users\ChengXi\Desktop\cstnet2\2c101ed073993aec0fc35999dd807dfc_4_sketch.png', r'C:\Users\ChengXi\Desktop\cstnet2\2c101ed073993aec0fc35999dd807dfc_4_sketch.png')
    augment_folder("input_pngs", "output_pngs", ratio=1.5)





