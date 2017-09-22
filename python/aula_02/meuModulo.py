import math as m
def RMSLE(y_true, y_pred):
    """ Root Mean Squared Logarithmic Error 
    Parâmetros
    ----------
    y_true : 1d array
        Rótulos.
    y_pred : 1d array
        Predições, retornadas pelo modelo.
    """
    soma=0
    for idx, item in enumerate(y_pred):
        n1 = m.log(item)
        n2 = m.log(y_true[idx])
        soma = soma + (n1 - n2)**2
    return m.sqrt((soma/len(y_pred)))