# pack
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
# self
import utils


def remove_png_white_pixel(png_file):
    # 打开图片
    img = Image.open(png_file).convert("RGBA")

    # 获取像素数据
    pixels = img.getdata()

    # 创建一个新的像素列表
    new_pixels = []
    for pixel in pixels:
        # 检查像素是否为白色
        if pixel[:3] == (255, 255, 255):
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


def remove_png_white_pixel_batched(dir_path, workers=4):
    # 获取全部图片路径
    pictures_all = utils.get_allfiles(dir_path, 'png', False)

    with Pool(processes=workers) as pool:
        _ = list(
            tqdm(
                pool.imap(remove_png_white_pixel, pictures_all),
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



