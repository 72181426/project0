import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

R2 = X**2 + Y**2
Z = np.sin(R2) / R2
Z[R2 == 0] = 1

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X, Y, Z,
    cmap='jet',
    edgecolor='k',
    linewidth=0.25,
    alpha=0.88,          # <-- transparency
    antialiased=True
)

# Similar camera angle
ax.view_init(elev=18, azim=-50)

plt.show()