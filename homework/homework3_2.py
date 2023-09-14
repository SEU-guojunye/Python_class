from numpy import *
from matplotlib.pyplot import *
from numpy import polynomial as P  # 用于创建多项式
from numpy import linalg as L  # 用于线性方程组求解
from scipy.special import roots_legendre  # 用于获得高斯型积分所需参数
import time  # 代码计时
from matplotlib.animation import FuncAnimation  # 动画用

################################################################求解部分


def SolveODE(
    Basis, dBasis, x, w, NonLinearF, Alpha
):  # Basis,dBasis所取基及其导数 x，w为求积节点和权重（向量形式） NonLinearF alpha决定ODE
    n = size(Basis)
    b = zeros(n)  # 相乘所得向量
    A = zeros([n, n])  # 相乘所得对称矩阵
    f = NonLinearF(x)  # 右端f于求积节点处函数值

    templistBasis = [Basis[0](x)]
    templistdBasis = [dBasis[0](x)]
    flag = 1

    for i in range(0, n):
        b[i] = dot(templistBasis[i] * f, w)  # 数值积分
        for j in range(0, n):
            if j >= i:
                A[i, j] = Alpha * dot(templistBasis[i] * templistBasis[j], w) + dot(
                    templistdBasis[i] * templistdBasis[j], w
                )
            else:
                A[i, j] = A[j, i]  # 对称，节省计算量
            if flag:
                if j == n - 1:
                    flag = 0
                else:
                    templistBasis.append(Basis[j + 1](x))
                    templistdBasis.append(dBasis[j + 1](x))  # 避免反复运算基于基的导数在求积节点处的值

    return dot(L.solve(A, b), Basis)  # 内积结果为解函数


################################################################Jacobi求基函数
def JacobiSeq(N):
    Polynomialx = P.Polynomial([0, 1], domain=[-1, 1])
    PolynomialOne = P.Polynomial(1, domain=[-1, 1])
    Jacobi00 = [PolynomialOne]  # 多项式1，而非浮点数1#参数为00
    dJacobi00 = [P.Polynomial(0, domain=[-1, 1])]
    if N == 1:
        return array(Jacobi00), array(dJacobi00)  # 特殊处理
    Jacobi00.append(Polynomialx)
    dJacobi00.append(PolynomialOne)
    if N == 2:
        return array(Jacobi00), array(dJacobi00)  # 特殊处理
    Jacobi00.append(P.Polynomial([-0.5, 0, 1.5], domain=[-1, 1]))
    dJacobi00.append(P.Polynomial([0, 3], domain=[-1, 1]))
    if N == 3:
        return array(Jacobi00), array(dJacobi00)  # 特殊处理
    Jacobi11 = [PolynomialOne, P.Polynomial([0, 2], domain=[-1, 1])]  # 参数为11
    for n in range(2, N):  # n为当前最后一个Jacobi00的下标,注意边界
        JacobiA = (2 * n + 1) * (2 * n + 2) / (2 * ((n + 1) ** 2))
        JacobiC = (2 * n + 2) * (n**2) / (((n + 1) ** 2) * (2 * n))
        Jacobi00.append(
            (JacobiA * Polynomialx) * Jacobi00[n] - JacobiC * Jacobi00[n - 1]
        )
        JacobiA = (2 * n + 1) * (2 * n + 2) / (2 * n * (n + 2))
        JacobiC = (2 * n + 2) * (n**2) / ((n + 2) * 2 * (n**2))
        Jacobi11.append(
            (JacobiA * Polynomialx) * Jacobi11[n - 1] - JacobiC * Jacobi11[n - 2]
        )  # 内嵌，减少堆栈次数
        dJacobi00.append(((n + 2) * Jacobi11[n]) / 2)  # 注意为n+2，Jacobi11比Jacobi00少一项
    return array(Jacobi00), array(dJacobi00)  # 得到多项式，向量形式便于后续处理


################################################################直接求导


def CreateLegendreSequence(N):  # N为多项式个数
    PolynomialOne = P.Polynomial(1, domain=[-1, 1])
    LegendreSequence = [PolynomialOne]  # 多项式1，而非浮点数1
    dLegendreSequence = [P.Polynomial(0, domain=[-1, 1])]
    if N == 1:
        return array(LegendreSequence), array(dLegendreSequence)  # 特殊处理
    Polynomialx = P.Polynomial([0, 1], domain=[-1, 1])
    LegendreSequence.append(Polynomialx)
    dLegendreSequence.append(PolynomialOne)
    for n in range(1, N):  # n为当前最后一个多项式的下标
        LegendreSequence.append(
            (
                (2 * n + 1) * Polynomialx * LegendreSequence[n]
                - n * LegendreSequence[n - 1]
            )
            / (n + 1)
        )  # 往后添加第（n+1）个
        dLegendreSequence.append(
            P.Polynomial(
                (LegendreSequence[n + 1].coef * array(range(0, n + 2)))[1 : n + 2],
                domain=[-1, 1],
            )
        )  # 对最后一个多项式求导#内嵌，减少堆栈次数
    return array(LegendreSequence), array(dLegendreSequence)  # 得到多项式，向量形式便于后续处理


################################################################分析部分

xVals = linspace(-1, 1, 1000)  # 绘图用分点，全局变量


def TestAndDraw(
    Basis,
    dBasis,
    EXAGauss,
    NonLinearF,
    Alpha,
    N,
    GaussX,
    GaussW,
    X,
    tempEXA,
    nVals,
    flag,
    TestTimes,
):
    Solution = list()
    L2Solution = list()

    for n in range(1, N + 1):
        Solution.append(
            SolveODE(Basis[0:n], dBasis[0:n], GaussX, GaussW, NonLinearF, Alpha)
        )
        L2Solution.append(
            dot((Solution[n - 1](GaussX) - EXAGauss) ** 2, GaussW) ** (1 / 2)
        )  # L2意义下的差

    LogL2 = log(L2Solution)  # 取对数，后续进行一次多项式拟合
    Fitcurve = P.Polynomial.fit(nVals, LogL2, deg=1)
    Logcurve = Fitcurve(nVals)  # 生成拟合数据

    print(flag, "得到解", Solution[N - 1], "\n收敛阶为:", -Fitcurve.coef[1])

    figure()
    xlabel("x")
    ylabel("y")
    title("Solution")
    for m in range(0, N):
        plot(X, Solution[m](X), label=m)
    plot(X, tempEXA, label="exact")
    legend()
    show()

    M = max(tempEXA) + 0.1
    m = min(tempEXA) - 0.1

    fig, ax = subplots()  # 创建一个空白图形
    (line1,) = ax.plot(X, zeros_like(X), lw=2)  # 创建一个空白的线条对象
    text = ax.text(-1, M - 0.1, " ")

    def init():  # 初始化函数，不需要更新数据
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(m, M)
        ax.plot(X, tempEXA)
        return line1, text

    def update(frame):  # 更新函数，用于更新每一帧的数据
        i = frame
        line1.set_data(X, Solution[i // 10](X))
        text.set_text("N=" + str(i // 10))
        return line1, text

    ani = FuncAnimation(
        fig, update, frames=nVals * 10, init_func=init, repeat=False
    )  # 创建动画对象
    filename = flag + "animation" + str(TestTimes) + ".gif"
    ani.save(filename)
    show()  # 显示动画

    figure()
    title("n-log(error_L2)")
    xlabel("n")
    ylabel("log(error_L2)")
    plot(nVals, LogL2, label="ORG")
    plot(nVals, Logcurve, label="Fit")
    legend()
    show()


def TestControl(
    Fexact, NonLinearF, Alpha, N=10, GaussOrder=0, X=xVals, TestTimes=0
):  # 精确解，f,alpha，近似解所用项数的最大值，Gauss求积公式所用点数，绘图时X所取坐标
    if GaussOrder == 0:
        GaussOrder = N + 2  # 解的次数最高N+1次，要求对2*(N+1)次精确（点数为n,2*(n-1)+1>=2*(N+1)）
    GaussX, GaussW = roots_legendre(N + 2)
    GaussX, GaussW = array(GaussX), array(GaussW)
    tempEXA = Fexact(X)
    EXAGauss = Fexact(GaussX)  # 减少计算次数
    nVals = array(list(range(0, N)))  # 绘图用

    timeBegin = time.time()
    LegendreSeq, dLegendreSeq = CreateLegendreSequence(N + 2)
    timeEnd = time.time()
    print("求导法求基耗时", timeEnd - timeBegin)
    BasisA, dBasisA = (
        LegendreSeq[0:N] - LegendreSeq[2 : N + 2],
        dLegendreSeq[0:N] - dLegendreSeq[2 : N + 2],
    )
    timeBegin = time.time()
    LegendreSeq, dLegendreSeq = JacobiSeq(N + 2)
    timeEnd = time.time()
    print("Jacobi法求基耗时", timeEnd - timeBegin)
    BasisB, dBasisB = (
        LegendreSeq[0:N] - LegendreSeq[2 : N + 2],
        dLegendreSeq[0:N] - dLegendreSeq[2 : N + 2],
    )  # 只生成一次基，避免重复运算

    TestAndDraw(
        BasisA,
        dBasisA,
        EXAGauss,
        NonLinearF,
        Alpha,
        N,
        GaussX,
        GaussW,
        X,
        tempEXA,
        nVals,
        "求导法",
        TestTimes,
    )
    TestAndDraw(
        BasisB,
        dBasisB,
        EXAGauss,
        NonLinearF,
        Alpha,
        N,
        GaussX,
        GaussW,
        X,
        tempEXA,
        nVals,
        "Jacobi法",
        TestTimes,
    )

    return TestTimes + 1


################################################################测试部分

AllTestTimes = 1

NonLinearF = lambda x: x**10 - 1 - 90 * (x**8)  # 定义非线性项f
ExactU = lambda x: x**10 - 1  # 精确解
AllTestTimes = TestControl(ExactU, NonLinearF, 1, TestTimes=AllTestTimes)
# 基包含10次项后精度突然升高

####################

NonLinearF = lambda x: pi**2 * sin(pi * x)
ExactU = lambda x: sin(pi * x)
AllTestTimes = TestControl(ExactU, NonLinearF, 0, TestTimes=AllTestTimes)

####################

NonLinearF = lambda x: (4 + 4 * x**2) * sin(x**2 - 1) - 2 * cos(x**2 - 1)
ExactU = lambda x: sin(x**2 - 1)
AllTestTimes = TestControl(ExactU, NonLinearF, 4, TestTimes=AllTestTimes)

####################

NonLinearF = lambda x: 2 * (x**2) - 4
ExactU = lambda x: x**2 - 1
AllTestTimes = TestControl(ExactU, NonLinearF, 2, TestTimes=AllTestTimes)
# 误差太小，负的收敛阶应为浮点运算误差导致
