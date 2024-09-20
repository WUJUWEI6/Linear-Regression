import numpy as np
import matplotlib.pyplot as plt

class Sin_regression:
    def __init__(self, learn_speed_rate, echo_times):
        self.speed = learn_speed_rate   # 学习率
        self.times = echo_times         # 迭代次数
        self.c = np.random.rand(6)      # 随机初始化参数，避免局部最优
    
    # 梯度下降求最小损失时的参数
    def fit(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        c = self.c

        for idx in range(1, self.times + 1):
            # 预测序列
            predict_Y = c[0] + c[1] * X + c[2] * X ** 2 + c[3] * X ** 3 + c[4] * X ** 4 + c[5] * X ** 5

            # 每100次打印损失函数值
            if idx % 100 == 0:
                print(f"time: {idx}, loss: {self.loss(predict_Y, Y):.4f}")

            # 计算偏导
            temp = np.zeros(6)
            for i in range(6):
                temp[i] = np.mean(2 * (predict_Y - Y) * (X ** i))
            
            # 按梯度反方向更新参数
            self.c -= self.speed * temp

        return self.c
    
    def loss(self, predict_Y: np.ndarray, Y: np.ndarray) -> float:
        return np.mean((predict_Y - Y) ** 2)

if __name__ == "__main__":
    # 生成训练集数据
    X = np.linspace(-np.pi, np.pi, 100)
    Y = np.sin(X)

    # 加载模型
    model = Sin_regression(learn_speed_rate = 0.0001, echo_times = 10000)
    c = model.fit(X, Y)
    prediction_Y = c[0] + c[1] * X + c[2] * X ** 2 + c[3] * X ** 3 + c[4] * X ** 4 + c[5] * X ** 5

    # 设置字体为支持中文的字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码

    # 可视化训练集和预测结果
    plt.figure(figsize = (10, 6))
    plt.title('sin函数的关于一元五次方程的拟合')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(X, Y, color = 'red', label = 'sin(x)')
    
    plt.plot(X, prediction_Y, linestyle='--', color='orange', label = '预测')
    plt.show()