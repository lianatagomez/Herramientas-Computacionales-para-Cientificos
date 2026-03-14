import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# TEXTO DE RESULTADOS
# --------------------------------------------------

def crear_tablas(all_results, tau):

    text = ""

    for res in all_results:

        beta = res["beta"]
        S = res["S"]
        J = res["J"]
        I_surface = res["I_surface"]

        text += "\n====================================\n"
        text += f"Resultados para beta = {beta}\n"
        text += "tau            S(tau)          J(tau)\n"
        text += "------------------------------------\n"

        for t, s, j in zip(tau, S, J):
            text += f"{t:10.4e}   {s:10.4e}   {j:10.4e}\n"

        text += "\nIntensidad emergente I(mu):\n"

        for i, val in enumerate(I_surface):
            text += f"mu[{i}] = {val:.6f}\n"

    return text


# --------------------------------------------------
# FIGURAS
# --------------------------------------------------

def crear_figuras(all_results, tau):

    # -------- S(tau) --------

    fig_S, ax_S = plt.subplots()

    for res in all_results:
        ax_S.plot(tau, res["S"], label=f"beta={res['beta']}")

    ax_S.set_xlabel("tau")
    ax_S.set_ylabel("S(tau)")
    ax_S.set_xscale("log")
    ax_S.set_title("Funcion fuente")
    ax_S.grid()
    ax_S.legend()

    return fig_S


# --------------------------------------------------
# GUARDAR DATOS
# --------------------------------------------------

def save_data_files(all_results, tau):

    for res in all_results:

        beta = res["beta"]
        S = res["S"]
        J = res["J"]

        data = np.column_stack((tau, S, J))

        filename = f"feautrier_results_beta_{beta}.txt"

        np.savetxt(
            filename,
            data,
            header="tau   S(tau)   J(tau)"
        )


# --------------------------------------------------
# GUARDAR FIGURAS
# --------------------------------------------------

def save_figures(fig_S):

    fig_S.savefig("S_vs_tau.png", dpi=300)
    