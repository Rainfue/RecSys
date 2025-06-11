# Импортирование библиотек
# -------------------------------------------------------------------
from math import log        # Импортирую логарифм 
from math import e
from random import randint
from tqdm import tqdm


# Построение класса с моделью
# -------------------------------------------------------------------
class LogisticRegression():
    # Конструктор класса
    def __init__(self,
                 inputs: list,
                 targets: list,
                 epochs: int = 50,
                 lr: float = 0.01
                 ):
        
        self.inputs = inputs
        self.targets = targets
        self.epochs = epochs
        self.lr = lr
        # m - число весов
        self.m = len(inputs[0]) + 1
        self.weights = [randint(-100, 100)/100 for _ in range(self.m)]

        # self.lamb = 10 ** np.random.uniform(-5, 2)
        self.lamb = 0.000001
        
    # -----------------------------------------------
    # Взвешеная сумма
    def weighted_z(self, point):
        z = [item * self.weights[i] for i, item in enumerate(point)]
        return sum(z) + self.weights[-1]

    # -----------------------------------------------
    # Логитистическая ф-я (ответ модели)
    def logistic_function(self, z):
        return 1 / (1+e**(-z))
    
    # -----------------------------------------------
    # L2 регуляризация 
    def l2_reg(self, target, output):
        error = -(target*log(output, e)+(1-target)*log(1-output, e))
        error += self.lamb*sum([self.weights[i]**2 for i in range(self.m-1)])
        return error
    
    # -----------------------------------------------
    # Ф-я логистической ошибки
    def logistic_error(self):
        errors = []

        for i, point in enumerate(self.inputs):
            z = self.weighted_z(point)
            output = self.logistic_function(z)
            target = self.targets[i]

            # Обработка исключений, т.к. логарифм 0 не определен
            if output == 1:
                output = 0.99999
            if output == 0:
                output = 0.00001
            
            error = self.l2_reg(target, output)
            errors.append(error)

        return sum(errors) / len(errors)
    
    # -----------------------------------------------
    # Ф-я для нахождения статистики предсказаний
    def get_stats(self, outputs, targets):
        tp, tn, fp, fn = 0, 0, 0, 0
        for i in range(len(outputs)):
            if round(outputs[i]) == targets[i]:
                if targets[i] == 1:
                    tp += 1
                if targets[i] == 0:
                    tn += 1
            else:
                if targets[i] == 1:
                    fp += 1
                if targets[i] == 0:
                    fn += 1

        return tp, tn, fp, fn
    
    # -----------------------------------------------
    # Ф-я для accuracy
    def accuracy(self, tp, tn, fp, fn):
        return (tp+tn) / (tp+tn+fp+fn) 
    
    # -----------------------------------------------
    # Ф-я для precision
    def precision(self, tp,fp):
        return tp / (tp+fp)

    # -----------------------------------------------
    # Ф-я для recall
    def recall(self, tp, fn):
        return tp / (tp+fn)

    # -----------------------------------------------
    # Ф-я для F1-score
    def f1_score(self, precision, recall):
        return 2 * (precision * recall)/(precision + recall)

    # Функция для обучения модели
    def train(self):
        '''Ф-я для обучения модели на своих данных'''
        
        outputs = []
        targets = []

        progress_bar = tqdm(range(self.epochs), desc="Training Progress", unit="epoch")

        for epoch in progress_bar:
            for i, point in enumerate(self.inputs):
                z = self.weighted_z(point)
                output = self.logistic_function(z)
                target = self.targets[i]

                outputs.append(output)
                targets.append(target)

                for j in range(self.m - 1):
                    self.weights[j] -= self.lr * (
                        (output - target) * point[j] + self.lamb * self.weights[j]
                    )
                self.weights[-1] -= self.lr * (output - target)

            tp, tn, fp, fn = self.get_stats(outputs, targets)

            accuracy = round(self.accuracy(tp, tn, fp, fn), 4)
            precision = round(self.precision(tp, fp), 4)
            recall = round(self.recall(tp, fn), 4)
            f1_score = round(self.f1_score(precision, recall), 4)

            progress_bar.set_postfix({
                'error': f'{self.logistic_error():.4f}',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1 score': f1_score,
            })

            outputs = []
            targets = []

        print(f'Финальные веса: {self.weights}')

    def visualize_predicts(self):
        outputs = []
        for i, point in enumerate(self.inputs):
            z = self.weighted_z(point)
            output = self.logistic_function(z)
            outputs.append(output)

        # Разделяем точки на два класса (0 и 1) по порогу 0.5
        class_0 = [point for point, output in zip(self.inputs, outputs) if output < 0.5]
        class_1 = [point for point, output in zip(self.inputs, outputs) if output >= 0.5]

        # Визуализация
        plt.figure(figsize=(8, 6))
        
        # Точки класса 0 (красные)
        if class_0:
            x0, y0 = zip(*class_0)
            plt.scatter(x=x0, y=y0, color='red', label='Class 0 (Predicted)')
        
        # Точки класса 1 (синие)
        if class_1:
            x1, y1 = zip(*class_1)
            plt.scatter(x=x1, y=y1, color='blue', label='Class 1 (Predicted)')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Model Predictions')
        plt.legend()
        plt.grid(True)
        plt.savefig("predicted_plot.png", dpi=300)
        
# -------------------------------------------------------------------

# Тестирование модели на синтетических данных
# -------------------------------------------------------------------
if __name__ == '__main__':
    # Импортированиие библиотек
    # ---------------------------------------------------------
    from random import randint          # Для генерации случайных чисел
    import matplotlib.pyplot as plt     # Для визуализации
    from random import shuffle

    # Функция
    # ---------------------------------------------------------
    def data_points(n_samples: int, class_point: tuple, noise: int | float = 1):
        def offset_point():
            # Создаем случайные значения для разности точек в классе
            offset_x = randint(-100 * noise, noise * 100) / 100
            offset_y = randint(-100 * noise, noise * 100) / 100

            # Распределяем точку вокруг классового центра,
            # прибавляя случайные сгенерированные числа к центру класса
            x = class_point[0] + offset_x
            y = class_point[1] + offset_y

            # Возвращаем точку
            return x, y

        # Получаем список точек (кортежей - (x, y))
        points = [offset_point() for _ in range(n_samples)]
        # Разделяем список на список x и y
        x_list = [points[i][0] for i in range(n_samples)]
        y_list = [points[i][1] for i in range(n_samples)]

        # Возвращаем список x, и список y
        return x_list, y_list

    # Генерация набора данных
    # ---------------------------------------------------------
    x1_list, y1_list = data_points(n_samples=50, class_point=(1, 1), noise=0.9)
    x2_list, y2_list = data_points(n_samples=50, class_point=(4, 2.5), noise=2.5)
    

    # Вывод первых 5 точек
    print(x1_list[:5], y1_list[:5])
    print(x2_list[:5], y2_list[:5])

    # Визуализация
    # ---------------------------------------------------------
    plt.scatter(x=x1_list, y=y1_list, color='red')
    plt.scatter(x=x2_list, y=y2_list, color='blue')
    plt.savefig("plot.png", dpi=300)


    inputs = [(x1_list[i], y1_list[i]) for i in range(len(x1_list))]
    targets = [0 for _ in range(len(x1_list))]
    # Добавляем второй класс
    inputs += [(x2_list[i], y2_list[i]) for i in range(len(x2_list))]
    targets += [1 for _ in range(len(x2_list))]

    combined = list(zip(inputs, targets))
    shuffle(combined)
    inputs, targets = zip(*combined)

    split_idx = int(0.8 * len(inputs))  # 80% данных — на обучение

    # Обучающая выборка
    train_inputs = inputs[:split_idx]    
    train_targets = targets[:split_idx]

    # Тестовая выборка
    test_inputs = inputs[split_idx:]
    test_targets = targets[split_idx:]

    model = LogisticRegression(inputs, targets, epochs=100)
    model.train()
    model.visualize_predicts()