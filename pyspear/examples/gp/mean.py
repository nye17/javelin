from pyspear.gp import *

# Generate mean
def quadfun(x, a, b, c):
    return (a * x ** 2 + b * x + c)

def const(x):
    return(2.*(x*0.0+1.0))

M  = Mean(quadfun, a = 1., b = .5, c = 2.)
M2 = Mean(const)

#### - Plot - ####
if __name__ == '__main__':
    from pylab import *
    x=arange(-1.,1.,.1)

    clf()
    plot(x, M(x), 'k-')
    plot(x, M2(x), 'k-')
    xlabel('x')
    ylabel('M(x)')
    axis('tight')
    show()
