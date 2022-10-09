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


def hash_tensor(tensor):
    return hash(" ".join([str(value) for value in tensor.flatten()]))


def model_hashes_identical(a, b):
    hashes_a = [hash_tensor(p) for p in a.parameters()]
    hashes_b = [hash_tensor(p) for p in b.parameters()]
    return set(hashes_a) == set(hashes_b)
