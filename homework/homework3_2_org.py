from numpy import *
from matplotlib.pyplot import *
from numpy import polynomial as P
from numpy import linalg as L#线性方程组求解
from scipy.special import roots_legendre#用于获得高斯型积分所需参数
import time  # 代码计时

def createb(Basis,f,x,w):#x，w为求积节点和权重
    n=size(Basis)
    b=zeros(n)#相乘所得向量
    for i in range(0,n):
        b[i]=dot((Basis[i](array(x)))*(f(array(x))),array(w))#数值积分
        if abs(b[i])<0.00000001:
            b[i]=0
    return b

def createA(f,x,w):
    n=size(f)
    A=zeros([n,n])#相乘所得对称矩阵
    for i in range(0,n):
        for j in range(0,n):
            if j>=i:
                A[i,j]=dot(f[i](array(x))*f[j](array(x)),array(w))
                if abs(A[i,j])<0.00000001:
                    A[i,j]=0
            else:
                A[i,j]=A[j,i]
    return A

def SolveODE(Basis,dBasis,x,w,f,Alpha):#N：所取基的个数 x，w为求积节点和权重 f alpha决定ODE
    A=(Alpha*createA(Basis,x,w)+createA(dBasis,x,w))
    b=createb(Basis,f,x,w)
    #print(A,b)
    #print(L.solve(A,b))
    return dot(L.solve(A,b),Basis)#内积结果为解函数

################################################################Jacobi求基函数

def JacobiA(a,b,n):
    return ((2*n+a+b+1)*(2*n+a+b+2))/(2*(n+1)*(n+a+b+1))

def JacobiB(a,b,n):
    return ((2*n+a+b+1)*(b**2-a**2))/(2*(n+1)*(n+a+b+1)*(2*n+a+b))

def JacobiC(a,b,n):
    return ((2*n+a+b+2)*(n+a)*(n+b))/((n+1)*(n+a+b+1)*(2*n+a+b))

def NextJacobi(a,b,n,JacobiSequence):
    return (JacobiA(a,b,n)*P.Polynomial([0,1],domain=[-1,1])-JacobiB(a,b,n))*JacobiSequence[n]-JacobiC(a,b,n)*JacobiSequence[n-1]

def JacobiSeq(N):
    Jacobi00=[P.Polynomial(1,domain=[-1,1])]#多项式1，而非浮点数1#参数为00
    dJacobi00=[P.Polynomial(0,domain=[-1,1])]
    if N==1:
        return Jacobi00,dJacobi00#特殊处理
    Jacobi00.append(P.Polynomial([0,1],domain=[-1,1]))
    dJacobi00.append(P.Polynomial(1,domain=[-1,1]))
    if N==2:
        return Jacobi00,dJacobi00#特殊处理
    Jacobi00.append(NextJacobi(0,0,1,Jacobi00))
    dJacobi00.append(P.Polynomial([0,3],domain=[-1,1]))
    if N==3:
        return Jacobi00,dJacobi00#特殊处理
    Jacobi11=[P.Polynomial(1,domain=[-1,1]),P.Polynomial([0,2],domain=[-1,1])]#参数为11
    for n in range(3,N):
        Jacobi00.append(NextJacobi(0,0,n-1,Jacobi00))
        Jacobi11.append(NextJacobi(1,1,n-2,Jacobi11))
        dJacobi00.append(((n+1)*Jacobi11[n-1])/2)
    return array(Jacobi00),array(dJacobi00)#得到多项式，向量形式便于后续处理

def BasisANDdBasisSeq(N):
    LegendreSeq,dLegendreSeq=JacobiSeq(N+2)
    return LegendreSeq[0:N]-LegendreSeq[2:N+2],dLegendreSeq[0:N]-dLegendreSeq[2:N+2]#基与基的导数

################################################################直接求基函数

def DiffPoly(f):
    CoefofF=f.coef
    n=size(CoefofF)
    return P.Polynomial((CoefofF*array(range(0,n)))[1:n],domain=[-1,1])#对多项式f求导 #可用Fx.deriv()

def CreateLegendreSequence(N):#N为多项式个数
    LegendreSequence=[P.Polynomial(1,domain=[-1,1])]#多项式1，而非浮点数1
    if N==1:
        return LegendreSequence#特殊处理
    Polynomialx=P.Polynomial([0,1],domain=[-1,1])
    LegendreSequence.append(Polynomialx)
    n=1#n为当前最后一个多项式的下标
    while n<N-1:#注意边界
        LegendreSequence.append(((2*n+1)*Polynomialx*LegendreSequence[n]-n*LegendreSequence[n-1])/(n+1))#往后添加第（n+1）个
        n+=1
    return array(LegendreSequence)#得到多项式，向量形式便于后续处理

def CreateBasisANDdbasis(N):
    LegendreSeq=CreateLegendreSequence(N+2)
    Basis=LegendreSeq[0:N]-LegendreSeq[2:N+2]
    dBasis=list()
    for phi in Basis:
        dBasis.append(DiffPoly(phi))
    return Basis,dBasis#基与基的导数

################################################################

N=10#“最大”的N,解的次数最高N+2
GaussX,GaussW=roots_legendre(N+2)#要求对N+2次精确（点数为n,2*(n-1)+1>N+2）
Alpha=1
def NonLinearF(x):
    return abs(x)-1#定义非线性项f
#精确解为exp(abs(x))-e alpha=1 f=-exp(1) ----- 合适
#精确解为x**2-1 alpha=1 f=x**2-3 ---- 理论上会得到精确解
#精确解为sin(pi*x) ---- 三角函数的多项式逼近为0

#print(GaussX,GaussW)

xVals=linspace(-1,1,1000)#x分点
#print(xVals)
################################################################
timeBegin = time.time()
BasisA,dBasisA=BasisANDdBasisSeq(N)
BasisB,dBasisB=CreateBasisANDdbasis(N)#只生成一次基，避免重复运算

SolutionA=list()
SolutionB=list()
SolutionC=list()

for n in range(1,N+1):
    SolutionA.append(SolveODE(BasisA[0:n],dBasisA[0:n],GaussX,GaussW,NonLinearF,Alpha))
    SolutionB.append(SolveODE(BasisB[0:n],dBasisB[0:n],GaussX,GaussW,NonLinearF,Alpha))

timeEnd = time.time()
print("耗时", timeEnd - timeBegin)