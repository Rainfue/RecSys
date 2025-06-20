from math_func import scalar

# Функция для вычисления косинусного сходства
def cosine_similarity(x: list, y: list) -> float:
    if len(x) != len(y):
        raise TypeError('Вектора должны быть одинаковой длины')

    scalar_multiplication = sum([x[i]*y[i] for i in range(len(y))])
    return scalar_multiplication / (scalar(x) * scalar(y))