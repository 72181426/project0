import geopandas as gpd
import matplotlib.pyplot as plt

# Cargar el archivo local
states = gpd.read_file("brasil_estados.geojson")

# Graficar
fig, ax = plt.subplots(figsize=(10, 10))
states.plot(ax=ax, edgecolor="black", facecolor="#e0e0e0")

ax.set_title("Mapa de Brasil con sus estados", fontsize=16)
ax.axis("off")

plt.show()
