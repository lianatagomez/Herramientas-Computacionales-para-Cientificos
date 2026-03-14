"""
feautrier_class.py


"""

import numpy as np
from scipy import linalg


class Feautrier:
    """
  
    """

    def __init__(self, beta_value, mu, w, tau=None):
        """
        Inicializa los parámetros físicos del problema, la cuadratura y la grilla en profundidad ópticas.
        Vamos a dejar como predeterminado la grilla de 31 valores de tau. En el
        caso en querramos pasar otra grilla sólo hay que cambiar el valor de p_values.
        """

        # -------------------------------------------------
        # Parámetros físicos
        # -------------------------------------------------

        self.beta = beta_value
        self.alfa = 1.0 - self.beta


        # -------------------------------------------------
        # Cuadratura de Gauss. Se recibe desde función externa los valores de mu y w.
        # Tenemos dos opciones: orden 2 y orden 4.
        # -------------------------------------------------
        self.mu = mu                       #raíces de los polinomios de Legendre
        self.w = w                         #pesos correspondientes
        self.N_mu = len(self.mu)    

        # -------------------------------------------------
        # Construcción de la grilla en profundidad óptica
        # 10^-3, 10^-2.8, 10^-2.6,..., 1,..., 10^2.6, 10^2.8, 10^3
        # Entonces son 15 puntos a ambos lados del 1. Tengo 31 puntos
        # -------------------------------------------------

        if tau is None:
            self.p_values = np.arange(-3.0, 3.2, 0.2)
            self.tau = 10 ** self.p_values
        else:
            self.tau = tau
        
        self.N_tau = len(self.tau)
        
        # -------------------------------------------------
        # Inicialización de matrices del sistema
        # -------------------------------------------------

        # Matrices del sistema. Tengo N_tau matrices A de dimensión muXmu.
        # Lo mismo para B y C. Estas matrices son los "coeficientes" que acompañan
        # a los vectores U. Salen de hacer diferencias finitas sobre la ec diferencial.

        self.A = np.zeros((self.N_tau, self.N_mu, self.N_mu))
        self.B = np.zeros_like(self.A)
        self.C = np.zeros_like(self.A)

        # vector de betas, N_tau vectores L de dimensioń N_mu
        self.L = np.zeros((self.N_tau, self.N_mu))

        # matrices auxiliares que usamos para obtener U
        self.V = np.zeros_like(self.L)
        self.D = np.zeros_like(self.A)

        # solución del sistema
        self.U = np.zeros_like(self.L)

        # cantidades físicas derivadas
        self.J = np.zeros(self.N_tau)
        self.S = np.zeros(self.N_tau)

        # intensidad emergente
        self.I_surface = np.zeros(self.N_mu)

    # =====================================================
    # CONSTRUCCIÓN DEL SISTEMA DE ECUACIONES
    # =====================================================

    def calculo_hacia_adelante(self):    
        """
        Construye las matrices A, B y C  que surgen de la
        discretización de la ecuación de Feautrier en cada capa de la atmósfera.
        """

        for d in range(self.N_tau):        

            # -------------------------------------------------
            # BORDE EXTERNO
            # -------------------------------------------------
            # se implementa la condición de borde superior del problema
            # correspondiente a ausencia de radiación incidente externa
            if d == 0:

                t = self.tau[1] - self.tau[0]                 #delta tau_3/2 = tau2 - tau1
 
                self.C[d, :, :] = -(self.mu ** 2) / t ** 2    #escribe los elementos de la matriz C y B
                self.B[d, :, :] = self.alfa * self.w / 2      #Tengo una fila por vector mu.

                np.fill_diagonal(                             #Lo anterior también da valores a la diagonal, pero no queremos 
                    self.B[d, :, :],                          #esos valores. Por eso acá escribimos la diagonal como nos conviene
                    -(self.mu ** 2) / t ** 2 - self.mu / t - 0.5 + self.alfa * self.w / 2
                )

                self.L[d, :] = -self.beta / 2

            # -------------------------------------------------
            # PUNTOS INTERNOS
            # -------------------------------------------------
            # 
            elif 0 < d < self.N_tau - 1:     #no incluye los extremos, o sea tau_0 y tau_30

                t_neg = self.tau[d] - self.tau[d - 1]
                t_pos = self.tau[d + 1] - self.tau[d]
                t = (t_pos + t_neg) / 2

                self.B[d, :, :] = self.alfa * self.w

                np.fill_diagonal(
                    self.B[d, :, :],
                    -(self.mu ** 2 / t) * (1 / t_pos + 1 / t_neg)
                    - 1
                    + self.alfa * self.w,
                )

                np.fill_diagonal(
                    self.A[d, :, :],
                    -(self.mu ** 2) / (t * t_neg),
                )

                np.fill_diagonal(
                    self.C[d, :, :],
                    -(self.mu ** 2) / (t * t_pos),
                )

                self.L[d, :] = -self.beta

            # -------------------------------------------------
            # BORDE INTERNO
            # -------------------------------------------------
            # condición de borde que es ... 
            elif d == self.N_tau - 1: #es decir, para tau_30

                t = self.tau[d] - self.tau[d - 1]

                self.B[d, :, :] = self.alfa * self.w / 2

                np.fill_diagonal(
                    self.B[d, :, :],
                    -(self.mu ** 2) / t ** 2 - self.mu / t - 0.5 + self.alfa * self.w / 2
                )

                np.fill_diagonal(
                    self.A[d, :, :],
                    -(self.mu ** 2) / t ** 2,
                )

                self.L[d, :] = -self.beta / 2 - self.mu / t

    # =====================================================
    # RESOLUCIÓN DEL SISTEMA
    # =====================================================

    def calculo_hacia_atras(self):
        """
        Resuelve el sistema calculando las matrices auxiliares V y D 
        """

        # -------------------------------------------------
        # Paso inicial de la relación de recurrencia  Ec 6.54. Uso dos formas de resolver
        # con linalg y con np.linalg para ver qué cambia 
        # -------------------------------------------------

        self.V[0] = -linalg.solve(self.B[0], self.L[0])
        self.D[0] = linalg.solve(self.B[0], self.C[0])

        # -------------------------------------------------
        # Con la relación de recurrencia y los valores de V0 y D0 puedo obtener las
        # demás matrices Vd y Dd ec 6.57. 
        # -------------------------------------------------

        for d in range(1, self.N_tau):     #va de 1 a 30, recordar N_tau=31

            E = self.B[d] - self.A[d] @ self.D[d - 1]  #auxiliar
            E_inv = np.linalg.inv(E)

            self.D[d] = E_inv @ self.C[d]

            self.V[d] = E_inv @ (self.L[d] + self.A[d] @ self.V[d-1])
        # -------------------------------------------------
        # Obtenemos Ud empezando de la última capa más interna 
        # hacia el exterior. 
        # -------------------------------------------------

        # solución en la última capa U30 ec 6.59
        self.U[self.N_tau - 1] = self.V[self.N_tau - 1]

        # reconstrucción de la solución en las capas superiores con ec 6.56
        # U29 en adelante 
        for d in range(self.N_tau - 2, -1, -1):
            self.U[d] = self.D[d] @ self.U[d + 1] + self.V[d]

        # -------------------------------------------------
        # Cálculo de la intesidad media J_tau
        # -------------------------------------------------

        self.J = 0.5 * np.sum(self.w * self.U, axis=1) #por cada tau suma sobre todos los mu (axis=1)

        # -------------------------------------------------
        # Cálculo de la función fuente S_tau
        # -------------------------------------------------

        self.S = self.alfa * self.J + self.beta

        # -------------------------------------------------
        # Obtenemos I a partir de la sol formal de la ecuación de trasporte. Ecuación .. 
        # -------------------------------------------------
        # 
        # utilizando la regla del trapecio

        for d in range(self.N_tau - 1):

            f1 = self.S[d] * np.exp(-self.tau[d] / self.mu) / self.mu
            f2 = self.S[d + 1] * np.exp(-self.tau[d + 1] / self.mu) / self.mu

            dt = self.tau[d + 1] - self.tau[d]

            self.I_surface += 0.5 * (f1 + f2) * dt

    # =====================================================
    # MÉTODOS GETTER
    # =====================================================

    def get_funcion_fuente(self):
        """Devuelve la función fuente calculada en cada capa."""
        return self.S

    def get_intensidad_especifica(self):
        """Devuelve la intensidad específica emergente en la superficie."""
        return self.I_surface

    def get_tau(self):
        """Devuelve la grilla de profundidad óptica."""
        return self.tau

    def get_J(self):
        """ Devuelve la intensidad media calculada en cada capa"""
        return self.J