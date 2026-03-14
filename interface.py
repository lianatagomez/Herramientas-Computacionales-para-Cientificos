import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QComboBox, QTextEdit,
    QFileDialog, QCheckBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# módulos del proyecto
from solucion import run_solver
from results import crear_tablas, crear_figuras, save_data_files, save_figures


# --------------------------------------------------
# variable global para almacenar la grilla tau
# --------------------------------------------------

tau = None


# --------------------------------------------------
# función auxiliar para crear figuras vacías
# --------------------------------------------------

def create_empty_figure():
    return Figure()


# --------------------------------------------------
# clase de la interfaz
# --------------------------------------------------

class Interface(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("Solver de Feautrier")

        layout = QVBoxLayout()

        # -----------------------------
        # selector de cuadratura
        # -----------------------------

        layout.addWidget(QLabel("Orden de cuadratura"))

        self.quad_box = QComboBox()
        self.quad_box.addItems(["2", "4"])

        layout.addWidget(self.quad_box)

        # -----------------------------
        # cargar grilla externa
        # -----------------------------

        self.button_file = QPushButton("Cargar grilla tau")
        self.button_file.clicked.connect(self.load_file)

        layout.addWidget(self.button_file)

        layout.addWidget(
            QLabel("Si no se carga archivo se usa la grilla interna")
        )

        # -----------------------------
        # opciones para guardar
        # -----------------------------

        self.save_data = QCheckBox("Guardar tablas")
        self.save_plots = QCheckBox("Guardar gráficos")

        layout.addWidget(self.save_data)
        layout.addWidget(self.save_plots)

        # -----------------------------
        # botón ejecutar
        # -----------------------------

        self.button_run = QPushButton("Ejecutar solver")
        self.button_run.clicked.connect(self.run)

        layout.addWidget(self.button_run)

        # -----------------------------
        # caja de resultados numéricos
        # -----------------------------

        self.results_box = QTextEdit()
        layout.addWidget(self.results_box)

        # -----------------------------
        # gráficos
        # -----------------------------

        self.canvas_S = FigureCanvas(create_empty_figure())
        
        layout.addWidget(self.canvas_S)
    
        self.setLayout(layout)

    # --------------------------------------------------
    # cargar archivo de grilla
    # --------------------------------------------------

    def load_file(self):

        global tau

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar archivo de grilla",
            "",
            "Text files (*.txt)"
        )

        if filename:
            tau = np.loadtxt(filename)

    # --------------------------------------------------
    # ejecutar solver
    # --------------------------------------------------

    def run(self):

        global tau

        # leer orden de cuadratura
        order = int(self.quad_box.currentText())

        # ejecutar solver
        all_results, tau_solver = run_solver(order, tau)

        # -----------------------------
        # mostrar resultados 
        # -----------------------------

        text = crear_tablas(all_results, tau_solver)

        self.results_box.setText(text)

        # -----------------------------
        # crear figuras
        # -----------------------------

        fig_S = crear_figuras(all_results, tau_solver)

        # -----------------------------
        # mostrar figuras en la ventana
        # -----------------------------

        self.canvas_S.figure = fig_S
        self.canvas_S.draw()

        # -----------------------------
        # guardar archivos si se pidió
        # -----------------------------

        if self.save_data.isChecked():
            save_data_files(all_results, tau_solver)

        if self.save_plots.isChecked():
            save_figures(fig_S, fig_J)


# --------------------------------------------------
# ejecutar aplicación
# --------------------------------------------------

app = QApplication(sys.argv)

window = Interface()
window.show()

sys.exit(app.exec_())