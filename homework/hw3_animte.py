import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.integrate as spi

# calculate the legendre polyomial according to a recursive method
# n: the order of legendre polyomial
# x: the point you calcualte the legendre polynomial
def legendre(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2 * n - 1) * x * legendre(n - 1, x) - (n - 1) * legendre(n - 2, x)) / n
    
# calculate the coefficient in legendre approximation
# f: the function
# n: the highest order of the legendre polyomial
def legendre_appro_coff(f, n):

    lower_bound = -1
    upper_bound = 1

    A = np.zeros([n, n])
    b = np.zeros([n, 1])

    # calculate the elements in the matrix
    for i in range(n):
        b[i] = spi.quad(lambda x: f(x) * legendre(i, x), lower_bound, upper_bound)[0]
        for j in range(n):
            A[i, j] = spi.quad(lambda x: legendre(i, x) * legendre(j, x), lower_bound, upper_bound)[0]
    
    # solve the equation set
    c = np.linalg.solve(A, b)

    return c

# use the the coefficient in legendre approximation to calculate
# c: the coefficient in legendre approximation
# x: function points to calculate
def legendre_appro(c, x):

    y = np.zeros(len(x))
    for i in range(len(c)):
        y = y + c[i] * legendre(i, x)
    return y

f1 = lambda x: x**4
f2 = lambda x: np.sin(2*np.pi*x)

# 创建一个空白图形
fig, ax = plt.subplots()

# 初始化数据
x = np.linspace(-1, 1, 100)
range_max = 16

n = np.linspace(1, range_max, range_max)

# 创建一个空白的线条对象
line1, = ax.plot(x, np.zeros_like(x), lw=2, label='Legendre approximation')
text = ax.text(-1.2, 1.2, " ", fontsize=12, ha='center', va='center')
# 添加图例


# 初始化函数，不需要更新数据
def init():
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.rcParams['text.usetex'] = True
    ax.plot(x, f2(x), color='red', label=r'$ f(x) =  \sin(2\pi x)$')
    plt.title(r'$ f(x) = \sin(2\pi x)$')
    ax.legend()
    plt.tight_layout()
    return line1, text

# 更新函数，用于更新每一帧的数据
def update(frame):
    i = frame
    c1 = legendre_appro_coff(f2, i)
    y1 = f1(x)
    appr_y1 = legendre_appro(c1, x)
    line1.set_data(x, appr_y1)
    text.set_text("N =" + str(i))
    ax.legend()
    return line1, text

# 创建动画对象
ani = FuncAnimation(fig, update, frames=range(range_max), init_func=init, repeat=False)



# 保存动画为视频文件（需要安装ffmpeg等支持库）
# ani.save('f2.gif', writer='ffmpeg', fps=5)

# 显示动画（可选）
plt.show()
