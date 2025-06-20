from math_func import mean, std

# Функция для масштабирования данных
def data_scale(x = list) -> list:
    '''Ф-я для масштабирования данных'''
    # Среднее арифметическое
    x_mean = mean(x)
    
    # Среднеквадратичное отклонение
    x_std = std(x)

    return [(xi-x_mean) / x_std for xi in x]