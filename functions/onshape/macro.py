import os

# 米换算到英寸需要乘的系数
METER_TO_IN = 39.3701

# 英寸换算到米需要乘的系数
IN_TO_METER = 0.0254

# 全局的单位 ['METER', 'in']
GLOBAL_UNIT = 'in'

# 一些中间文件的保存文件夹
# SAVE_ROOT = r'E:\document\DeeplearningIdea\multi_cmd_seq_gen\four_type_ofs'
# SAVE_ROOT = r'D:\document\DeeplearningIdea\multi_cmd_seq_gen\multi_sketch_prim'
# SAVE_ROOT = r'D:\document\DeeplearningIdea\multi_cmd_seq_gen\more_diverse_v1'
# SAVE_ROOT = r'D:\document\DeeplearningIdea\multi_cmd_seq_gen\test_ERSL'
# SAVE_ROOT = r'D:\document\DeeplearningIdea\multi_cmd_seq_gen\ellipse'
SAVE_ROOT = r'D:\document\DeeplearningIdea\multi_cmd_seq_gen\more_diverse_v2'

# 测试的文件连接
# URL = 'https://cad.onshape.com/documents/f8d3a3b2ddfbc6077f810cbc/w/50c3f52b580a97326eb89747/e/a824129468cfbb9a5a7f6bd0'  # test_multi_cmd
# URL = 'https://cad.onshape.com/documents/f1c9542b7e95a9d78ba55c0f/w/76dded0e20d753ae1236e323/e/aa5f575d45eb5319e4f39885'  # test_multi_sketch_prim
# URL = 'https://cad.onshape.com/documents/bb6a98692284898a7fc69158/w/66bec0414acff16f65edeb74/e/5e43133f5eee3b516a1b8941'  # more_diverse_v1
# URL = 'https://cad.onshape.com/documents/0575624a7945226d240f1670/w/2f450ba4e367f000ff5f815d/e/4ed65794bcc62b014b25b8b2'  # test_ERSL
# URL = 'https://cad.onshape.com/documents/c5bbd3f741f95fa290774b8c/w/42b717e172b4e47d67acead6/e/03c03c8f59d106604452530b'  # ellipse
URL = 'https://cad.onshape.com/documents/8f8d14d25311423bb8afcc72/w/4cbd9843ad184d99979a3e3c/e/91bc9e0d43a430b13453802d'  # more_diverse_v2
os.makedirs(SAVE_ROOT, exist_ok=True)

