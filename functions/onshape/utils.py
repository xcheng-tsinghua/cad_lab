from functions.onshape import macro


def get_unit_trans_coff(unit, trans_to):
    """
    获取单位转换参数（乘数）
    :param unit: 类似 ('METER', 1)
    :param trans_to: ['METER', 'in'] 米和英寸
    :return:
    """
    mul_unit = unit[1]

    if unit[0] == trans_to:
        return mul_unit

    elif unit[0] == 'METER' and trans_to == 'in':
        return  macro.METER_TO_IN ** mul_unit

    elif unit[0] == 'in' and trans_to == 'METER':
        return  macro.IN_TO_METER ** mul_unit

    else:
        raise NotImplementedError







