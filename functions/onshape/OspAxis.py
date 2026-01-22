from functions.onshape.OspPoint import OspPoint


class OspAxis(object):
    """
    定义空间中的轴，需要空间中的一个点和一个非零向量
    """
    def __init__(self, loc_x, loc_y, loc_z, dir_x, dir_y, dir_z, is_2d):
        self.loc = OspPoint(loc_x, loc_y, loc_z, is_2d)
        self.dir = OspPoint(dir_x, dir_y, dir_z, is_2d)
        self.is_2d = is_2d

    def __call__(self, param: float):
        dest = self.loc + param * self.dir
        return dest

