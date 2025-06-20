# Функция для вычисления скалярного умножения двух векторов (длина вектора)
def scalar(x: list) -> float:
    '''Ф-я для вычисления скалярного умножения двух векторов'''
    
    return (sum([i * i for i in x]))**0.5

# Функция для вычисления среднего значения
def mean(x = list) -> float:
    '''Ф-я для вычисления среднего значения'''
    # Число наблюдений
    n = len(x)

    return sum(x) / n

# Функция для вычисления стандартного (среднеквадратичного отклонения)
def std(x = list):
    '''Ф-я для вычисления стандартного отклонения'''
    # Число наблюдений
    n = len(x)
    # Среднее 
    x_mean = mean(x)
    # Наблюдения без среднего в квадрате
    wo_mean = [(xi-x_mean)**2 for xi in x]

    return (sum(wo_mean) / n)**0.5

# Функция для вычисления ковариации
def cov(x: list, y: list, ddof: int = 1) -> float:
    '''
    **Функция для вычисления ковариации**
        Args:
            - x (list | np.ndarray): первый вектор
            - y (list | np.ndarray): второй вектор
            - ddof (int): смещенная (1) или не смещенная (0) оценка
        Returns:
            - значение ковариации между векторами (float)
    '''

    # Проверка на одноразмерность векторов
    if len(x) != len(y):
        raise ValueError(f'Вектора должны иметь 1 размер! ({len(x)} != {len(y)})')
    
    # Находим среднее у векторов
    x_mean = mean(x)
    y_mean = mean(y)

    return sum([(xi - x_mean)*(yi - y_mean) for xi, yi in zip(x, y)]) / (len(x)-ddof)

# Функция для вычисления дисперсии
def var(x: list, ddof: int = 1) -> float:
    '''
    Функция для вычисления дисперсии
        Args: 
            - x (list | np.ndarray): вычисляемый вектор
            - ddof (int = 1) delta degrees of freedom:
                0 - смещенная оценка
                1 - не смещенная оценка

        Returns:
            - значение дисперсии (float)
    '''
    # Вычисляем среднее арифметическое списка
    x_mean = mean(x)
    # Находим длину списка
    n = len(x)
    # Возвращаем дисперсию
    return sum([(xi-x_mean)**2 for xi in x]) / (n - ddof)

# Функция для получения ковариационной матрицы
def covmat(x: list, y: list, ddof: int = 1) -> list:
    '''Ф-я для получения ковариационной матрицы'''
    
    return [
        [var(x,ddof), cov(x,y,ddof)],
        [cov(x,y,ddof), var(y,ddof)]
    ]