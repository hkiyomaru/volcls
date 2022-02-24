from modules.layers import GradReverse


def reverse_grad(x, alpha: float = 1.0):
    return GradReverse.apply(x, alpha)
