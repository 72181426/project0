import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el mapa real de Brasil (archivo local)
states = gpd.read_file("brasil_estados.geojson")

# 2. Datos reales de tasas de homicidio (2022)
data = {
    "state": [
        "Acre", "Alagoas", "Amapá", "Amazonas", "Bahia", "Ceará", "Distrito Federal",
        "Espírito Santo", "Goiás", "Maranhão", "Mato Grosso", "Mato Grosso do Sul",
        "Minas Gerais", "Pará", "Paraíba", "Paraná", "Pernambuco", "Piauí",
        "Rio de Janeiro", "Rio Grande do Norte", "Rio Grande do Sul", "Rondônia",
        "Roraima", "Santa Catarina", "São Paulo", "Sergipe", "Tocantins"
    ],
    "homic_rate": [
        28.6, 37.9, 50.6, 38.8, 47.1, 35.5, 11.3,
        29.3, 25.2, 28.0, 29.3, 18.7,
        12.6, 36.9, 26.1, 22.7, 37.8, 25.0,
        27.9, 36.7, 18.8, 34.3,
        30.5, 14.0, 8.7, 34.8, 30.5
    ]
}

df = pd.DataFrame(data)

# 3. Unir el mapa con los datos
merged = states.merge(df, left_on="name", right_on="state", how="left")

# 4. Graficar el mapa de tasas de homicidio
fig, ax = plt.subplots(figsize=(12, 12))

merged.plot(
    column="homic_rate",
    cmap="Reds",
    linewidth=0.8,
    edgecolor="black",
    legend=True,
    legend_kwds={"label": "Tasa de homicidios por 100 000 habitantes"},
    ax=ax
)

ax.set_title("Tasa de homicidios por estado en Brasil (2022)", fontsize=18)
ax.axis("off")

plt.show()