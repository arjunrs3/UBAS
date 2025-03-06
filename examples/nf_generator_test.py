from UBAS.generators.neural_foil_generator import NFGenerator
import numpy as np
import matplotlib.pyplot as plt


def test_nf_2D():
    airfoil = [0.02, 0.4, 0.12]
    nf = NFGenerator(qoi="CM", airfoil=airfoil)
    Res = np.linspace(1000000, 100000000, 100)
    alphas = np.linspace(-5, 15, 100)
    Res_grid, alphas_grid = np.meshgrid(Res, alphas)
    input_X = np.c_[np.ravel(Res_grid), np.ravel(alphas_grid)]
    x, cm = nf.generate(input_X)
    cm_grid = np.reshape(cm, Res_grid.shape)
    fig, ax = plt.subplots()
    contour = plt.contourf(Res, alphas, cm_grid, cmap='viridis', vmin=np.min(cm_grid), vmax=np.max(cm_grid), levels=100)
    fig.colorbar(contour)
    ax.set_xlabel("Re")
    ax.set_ylabel("Alpha (deg)")
    plt.show()

if __name__ == "__main__":
    test_nf_2D()