from sympy import *
from sympy.abc import x, k, tau
from sympy.core.basic import Basic, S, C, sympify
from sympy.core.function import Function

init_printing(use_unicode=True, wrap_line=False, no_global=True)

class mygau(Function):
    is_real = True
    nargs = 1
    @classmethod
    def eval(cls, x):
        _val = exp(-(Abs(x))**2)
        return(_val)

class myexp(Function):
    is_real = True
    nargs = 1
    @classmethod
    def eval(cls, x):
        _val = exp(-Abs(x))
        return(_val)

H=Lambda(x, (sign(x)+1.0)/2.0)

class kepler(Function):
    is_real = True
    nargs = 1
    @classmethod
    def eval(cls, x):
#        _val = exp(-Abs(x)) + 2.0*exp(-(Abs(10.0*x))**2)
#        _val = exp(-Abs(x))*H(Abs(x)-0.2)+exp(0.8)*exp(-(Abs(5.0*x))**2)*H(0.2-Abs(x))
#        _val = exp(-Abs(x))*H(Abs(x)-0.2)+exp(-0.2)*H(0.2-Abs(x))
        _val = exp(-Abs(x)) - 0.5*exp(-(Abs(10.0*x))**2)
        return(_val)






if __name__ == "__main__":    
    import numpy as np
    import matplotlib.pyplot as plt
    xval = np.arange(0, 5, 0.01)
    cval1= np.zeros_like(xval)
    cval2= np.zeros_like(xval)
    cval3= np.zeros_like(xval)

    kval = np.power(10.0, np.arange(-2, 1, 0.01))
    pval1= np.zeros_like(kval)
    pval2= np.zeros_like(kval)
    pval3= np.zeros_like(kval)

    f1 = myexp(x)
    f2 = mygau(x)
    f3 = kepler(x)
    pprint(f1)
    pprint(f2)
    pprint(f3)

    g1 = fourier_transform(f1, x, k)
    g2 = fourier_transform(f2, x, k)
    g3 = fourier_transform(f3, x, k)
    pprint(g1)
    pprint(g2)
    pprint(g3)

    fig = plt.figure()
    ax  = fig.add_subplot(111)



    for i, xv in enumerate(xval):
        x = xv
        cval1[i] = eval(str(f1))
        cval2[i] = eval(str(f2))
        cval3[i] = eval(str(f3))

    ax.plot(xval, cval1, "k-")
    ax.plot(xval, cval2, "r-")
    ax.plot(xval, cval3, "b-")

    plt.show()


    fig = plt.figure()
    ax  = fig.add_subplot(111)

    for i, kv in enumerate(kval):
        k = kv
        pval1[i] = eval(str(re(g1)))
        pval2[i] = eval(str(re(g2)))
        pval3[i] = eval(str(re(g3)))
    ax.plot(kval, pval1, "k-")
    ax.plot(kval, pval2, "r-")
    ax.plot(kval, pval3, "b-")

    ax.set_xscale("log")
    plt.show()


#    func = kepler_exp2(x, tau)
#    pprint(fourier_transform(func, x, k))
