import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# 1. Cargar el GeoJSON simplificado (el que te di antes)
gdf = gpd.read_file("brasil_simplificado.geojson")

# 2. Tabla con tasas de homicidio por 100 000 habitantes
#    Datos aproximados basados en el Atlas da Violência / FBSP
data = {
    "state": [
        "Acre", "Alagoas", "Amapa", "Amazonas", "Bahia", "Ceara", "Distrito Federal",
        "Espirito Santo", "Goias", "Maranhao", "Mato Grosso", "Mato Grosso do Sul",
        "Minas Gerais", "Para", "Paraiba", "Parana", "Pernambuco", "Piaui",
        "Rio de Janeiro", "Rio Grande do Norte", "Rio Grande do Sul",
        "Rondonia", "Roraima", "Santa Catarina", "Sao Paulo", "Sergipe", "Tocantins"
    ],
    "homic_rate": [
        22.0, 37.3, 29.8, 27.3, 36.1, 35.0, 18.0,
        22.9, 30.0, 26.0, 31.0, 24.0,
        16.0, 32.0, 28.0, 19.0, 34.0, 21.0,
        24.0, 33.0, 17.0,
        27.0, 29.0, 11.0, 12.0, 32.0, 23.0
    ]
}

df = pd.DataFrame(data)

# 3. Unir con el GeoDataFrame por nombre
# En el GeoJSON, la columna se llama "name"
gdf_merged = gdf.merge(df, left_on="name", right_on="state", how="left")

# 4. Revisar si algún estado quedó sin datos
missing = gdf_merged[gdf_merged["homic_rate"].isna()]
if not missing.empty:
    print("Estados sin datos de homicidio (revisa nombres):")
    print(missing[["name"]])
else:
    print("Todos los estados tienen datos.")

# 5. Crear el mapa coroplético
fig, ax = plt.subplots(1, 1, figsize=(8, 10))

gdf_merged.plot(
    column="homic_rate",
    cmap="OrRd",  # Escala de naranjas-rojos adecuada para violencia
    linewidth=0.7,
    edgecolor="black",
    legend=True,
    legend_kwds={
        "label": "Tasa de homicidios por 100 000 habitantes",
        "orientation": "vertical",
        "shrink": 0.7
    },
    ax=ax
)

ax.set_title("Brasil – Tasa de homicidios por estado\n(por 100 000 habitantes, Atlas da Violência / FBSP)", fontsize=13)
ax.axis("off")
plt.tight_layout()
plt.show()