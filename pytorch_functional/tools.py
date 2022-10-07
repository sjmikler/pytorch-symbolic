#  Copyright (c) 2022 Szymon Mikler


def get_parameter_count(model):
    cnt = 0
    for param in model.parameters():
        cnt += param.shape.numel()
    return cnt


def get_parameter_shapes(model):
    shapes = []
    for param in model.parameters():
        shapes.append(tuple(param.shape))
    return shapes


def model_similar(a, b):
    if get_parameter_count(a) != get_parameter_count(b):
        return False

    if sorted(get_parameter_shapes(a)) != sorted(get_parameter_shapes(b)):
        return False
    return True
