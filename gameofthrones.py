# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt

# 1. Creamos el grafo
G = nx.DiGraph()

# 2. Definimos casas y personajes (ejemplo simplificado)
casas = [
    "Casa Stark",
    "Casa Lannister",
    "Casa Targaryen",
    "Casa Baratheon",
    "Casa Greyjoy"
]

personajes = {
    "Casa Stark": {
        "padres": ["Eddard Stark", "Catelyn Stark"],
        "hijos": ["Robb Stark", "Sansa Stark", "Arya Stark", "Bran Stark", "Rickon Stark", "Jon Snow (bastardo)"]
    },
    "Casa Lannister": {
        "padres": ["Tywin Lannister", "Joanna Lannister"],
        "hijos": ["Cersei Lannister", "Jaime Lannister", "Tyrion Lannister"]
    },
    "Casa Targaryen": {
        "padres": ["Aerys II Targaryen (Rey Loco)", "Rhaella Targaryen"],
        "hijos": ["Rhaegar Targaryen", "Viserys Targaryen", "Daenerys Targaryen"]
    },
    "Casa Baratheon": {
        "padres": ["Steffon Baratheon", "Cassana Estermont"],
        "hijos": ["Robert Baratheon", "Stannis Baratheon", "Renly Baratheon"]
    },
    "Casa Greyjoy": {
        "padres": ["Balon Greyjoy"],
        "hijos": ["Yara Greyjoy", "Theon Greyjoy"]
    }
}

# 3. Agregamos nodos y relaciones al grafo
for casa in casas:
    G.add_node(casa, tipo="casa")
    # Personajes de esa casa
    datos = personajes.get(casa, {})
    padres = datos.get("padres", [])
    hijos = datos.get("hijos", [])

    for p in padres:
        G.add_node(p, tipo="padre")
        G.add_edge(casa, p, relacion="pertenece")

    for h in hijos:
        G.add_node(h, tipo="hijo")
        G.add_edge(casa, h, relacion="pertenece")

    # Relaciones padres -> hijos
    if padres and hijos:
        for p in padres:
            for h in hijos:
                G.add_edge(p, h, relacion="hijo")

# 4. Posiciones de los nodos (layout de red)
pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)

# 5. Separamos nodos por tipo para darle estilo
nodos_casas = [n for n, d in G.nodes(data=True) if d.get("tipo") == "casa"]
nodos_padres = [n for n, d in G.nodes(data=True) if d.get("tipo") == "padre"]
nodos_hijos = [n for n, d in G.nodes(data=True) if d.get("tipo") == "hijo"]

plt.figure(figsize=(14, 10))
plt.axis("off")
plt.title("Mapa de ideas: Casas y personajes de Juego de Tronos", fontsize=16)

# Dibujamos aristas
nx.draw_networkx_edges(G, pos, alpha=0.4, arrowstyle="->", arrowsize=12)

# Dibujamos nodos con color diferente por tipo
nx.draw_networkx_nodes(G, pos,
                       nodelist=nodos_casas,
                       node_color="#ffcc00",
                       node_size=1600,
                       edgecolors="black",
                       linewidths=1.5,
                       label="Casas")

nx.draw_networkx_nodes(G, pos,
                       nodelist=nodos_padres,
                       node_color="#ff6666",
                       node_size=1100,
                       edgecolors="black",
                       linewidths=1.0,
                       label="Padres/Señores")

nx.draw_networkx_nodes(G, pos,
                       nodelist=nodos_hijos,
                       node_color="#66b3ff",
                       node_size=800,
                       edgecolors="black",
                       linewidths=1.0,
                       label="Hijos")

# Etiquetas
nx.draw_networkx_labels(G, pos, font_size=9)

# Leyenda
plt.legend(scatterpoints=1, fontsize=10, loc="upper left")

plt.tight_layout()
plt.show()