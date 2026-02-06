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

    elif unit_name == 'METER' and trans_to[0] == 'in':
        return  macro.METER_TO_IN ** power

    elif unit_name == 'in' and trans_to[0] == 'METER':
        return  macro.IN_TO_METER ** power

    elif unit_name == 'RADIAN' and trans_to[1] == 'DEGREE':
        return macro.RADIAN_TO_DEGREE ** power

    elif unit_name == 'DEGREE' and trans_to[1] == 'RADIAN':
        return macro.DEGREE_TO_RADIAN ** power

    else:
        raise NotImplementedError


def no_closure(func):
    """
    标记该函数不允许捕获任何外层局部变量
    """
    free_vars = func.__code__.co_freevars
    if free_vars:
        raise RuntimeError(
            f'Function {func.__qualname__} is declared no-closure, '
            f'but captures outer variables: {free_vars}'
        )
    return func



