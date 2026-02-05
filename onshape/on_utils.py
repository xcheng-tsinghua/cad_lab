from onshape import macro


def get_unit_trans_coff(unit_name, power, trans_to):
    """
    获取单位转换参数（乘数）
    :param unit_name: 类似 'METER'
    :param power: 次方，表示平方米还是立方米等。1, 2, 3
    :param trans_to: ['METER', 'in'] 米和英寸
    :return:
    """
    # 转换相同，不必转换
    if unit_name == trans_to:
        return 1.0

    elif unit_name == 'METER' and trans_to == 'in':
        return  macro.METER_TO_IN ** power

    elif unit_name == 'in' and trans_to == 'METER':
        return  macro.IN_TO_METER ** power

    else:
        raise NotImplementedError







