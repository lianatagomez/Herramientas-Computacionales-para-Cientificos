from Feautrier_clase import Feautrier
from quadrature import get_quadrature


def run_solver(order, tau):

    mu, w = get_quadrature(order)

    beta_values = [0.1, 0.01, 1e-4]

    all_results = []

    for beta in beta_values:

        if tau is None:
            solucion = Feautrier(beta, mu, w)
        else:
            solucion = Feautrier(beta, mu, w, tau)

        solucion.calculo_hacia_adelante()
        solucion.calculo_hacia_atras()

        tau_solver = solucion.get_tau()

        S = solucion.get_funcion_fuente()
        J = solucion.get_J()
        I = solucion.get_intensidad_especifica()

        all_results.append({
            "beta": beta,
            "S": S,
            "J": J,
            "I_surface": I
        })

    return all_results, tau_solver