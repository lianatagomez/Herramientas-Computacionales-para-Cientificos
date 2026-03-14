import numpy as np
from scipy.special import roots_legendre


def get_quadrature(order):
    """
    Devuelve los raíces mu y los pesos correspondientes w para la cuadratura
    de Gauss-Legendre, para integrar J en el intervalo [0,1]

    Parámetros: 
    order : int
        Puede ser 2 o 4 
    Devuelve: 
    arreglos de mu y w

 """ 

    # Scipy devuelve las raíces x y pesos wi de Gauss-Legendre en el intervalo [-1,1]
    x, wi = roots_legendre(order)

    #pasamos de [-1,1] a [0,1]
    mu=0.5*(x+1)
    w=0.5*wi

    return mu, w