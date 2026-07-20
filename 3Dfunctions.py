import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def construir_superficie(nombre):
    """Devuelve los datos de una superficie 3D para el nombre dado."""
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    if nombre == 'sincirc':
        Z = np.sin(np.sqrt(X**2 + Y**2))
        titulo = 'z = sin(sqrt(x² + y²))'
    elif nombre == 'cuadratica':
        Z = X**2 + Y**2
        titulo = 'z = x² + y²'
    elif nombre == 'sincos':
        Z = np.sin(X) + np.cos(Y)
        titulo = 'z = sin(x) + cos(y)'
    else:
        raise ValueError(f'Función no reconocida: {nombre}')

    return X, Y, Z, titulo


def graficar_funciones_3d():
    """Grafica varias superficies 3D interactivas y rotables con el mouse."""
    funciones = [('sincirc', 'superficie original'), ('cuadratica', 'función cuadrática'), ('sincos', 'función trigonométrica')]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[titulo for _, titulo in funciones] + [''],
        specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, None]],
    )

    for i, (nombre, _) in enumerate(funciones):
        X, Y, Z, titulo = construir_superficie(nombre)
        if i == 0:
            row, col = 1, 1
        elif i == 1:
            row, col = 1, 2
        else:
            row, col = 2, 1
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', name=titulo), row=row, col=col)

    fig.update_layout(
        title='Funciones 3D interactivas',
        width=1400,
        height=1000,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.update_scenes(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube', row=1, col=1)
    fig.update_scenes(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube', row=1, col=2)
    fig.update_scenes(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube', row=2, col=1)

    output_file = 'grafica_3d_interactiva.html'
    fig.write_html(output_file, auto_open=False)
    print(f'Gráfica guardada en {output_file}')
    fig.show()


def graficar_sinc_stationary():
    """Grafica la superficie sinc con el punto estacionario en (0, 0) usando matplotlib."""
    x = np.linspace(-3, 3, 180)
    y = np.linspace(-3, 3, 180)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.where(R == 0, 1.0, np.sin(R) / R)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(Z.min(), Z.max())
    facecolors = cmap(norm(Z))

    ax.plot_surface(
        X,
        Y,
        Z,
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0.15,
        antialiased=True,
        shade=False,
        edgecolor='k'
    )

    ax.plot([-3, 3], [0, 0], [1, 1], color='black', linewidth=3)
    ax.plot([0, 0], [-3, 3], [1, 1], color='black', linewidth=3)
    ax.text(0, 0, 1.05, '(0,0)', color='black', fontsize=12, horizontalalignment='center')

    ax.set_xticks([-3, -1, 1, 3])
    ax.set_yticks([-3, -1, 1, 3])
    ax.set_zticks([-0.5, 0, 0.5, 1.0, 1.2])
    ax.set_facecolor('white')

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    fig.text(0.12, 0.06, 'Figure : (0,0) is a stationary point of this f', fontsize=16, color='#3b4cc0')
    fig.suptitle('', fontsize=18)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_file = 'grafica_sinc_estacionaria.png'
    fig.savefig(output_file, dpi=150, facecolor='white')
    print(f'Gráfica guardada en {output_file}')
    plt.show()


if __name__ == '__main__':
    graficar_sinc_stationary()
