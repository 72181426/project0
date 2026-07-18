import numpy as np
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


if __name__ == '__main__':
    graficar_funciones_3d()
