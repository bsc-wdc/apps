import numpy as np
class Error(object):
    def __init__(self,digits,precision):
        self.en = []
        self.dn = 0.0
        self.dnAux = []
        self.p = 0.0
        self.n = 0
        self.digits = digits
        self.precision = precision

class AError(Error):
    from pycompss.api.task import task
    from decimal import Decimal
    @task(returns=Decimal)
    def error(self,xn1,x1,append=True):
        from decimal import Decimal,getcontext
        getcontext().prec = self.precision
        if append:
            self.en.append(Decimal(xn1-x1))
            self.n += 1
            return self.en[self.n-1]
        else:
            return Decimal(xn1-x1)
    @task(returns=Decimal,isModifier=False)
    def decimals(self,e1,e0):
        from decimal import Decimal
        return -self.p*(abs(e1/e0)).log10()

    def criteria(self,rho):
        from decimal import Decimal
        rho = Decimal(rho)
        self.p = Decimal((rho*rho) /(rho-Decimal(1.0)))
    @task(returns=int,isModifier=False)
    def converge(self,d):
        #return self.dn > self.digits
        return d > self.digits
    

class CError(Error):
    def __init__(self,digits,precision,alpha):
        Error.__init__(self,digits,precision)
        self.alpha = alpha
    def error(self,xn1,append=True):
        if append:
            self.en.append(xn1-self.alpha)
            self.n += 1
            return self.en[self.n-1]
        else:
            return xn1-self.alpha
    def decimals(self):
        self.dn = -np.log10(abs(self.en[self.n-1]))
    def criteria(self,rho):
        self.p = rho
    def converge(self):
        return self.dn > self.digits