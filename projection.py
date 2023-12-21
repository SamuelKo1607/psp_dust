import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['figure.dpi']= 200

from paths import psp_model_location
from paths import figures_location

# Create a new plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_proj_type('ortho')
ax.view_init(elev=0, azim=0, roll=0)


# Load the STL files and add the vectors to the plot
your_mesh = mesh.Mesh.from_file(psp_model_location)
ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors,
                                                   facecolors=r"#ff0000",
                                                   edgecolors="none",
                                                   antialiased=False))

# Add a unit sphere.
u = np.linspace(0, 2 * np.pi, 1000)
v = np.linspace(0, np.pi, 1000)

x = 5 + 1/np.sqrt(2*np.pi) * np.outer(np.cos(u), np.sin(v))
y = -4 + 1/np.sqrt(2*np.pi) * np.outer(np.sin(u), np.sin(v))
z = -4 + 1/np.sqrt(2*np.pi) * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='k', antialiased=False)

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
ax.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
ax.set_aspect('equal')
plt.axis('off')

# Now we can save it to a numpy array.
fig.canvas.draw()
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
data = np.reshape(data,(-1,3))
fig.show()
print(set(data))

