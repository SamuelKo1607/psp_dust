import numpy as np
from stl import mesh
from mpl_toolkits import mplot3d
import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['figure.dpi'] = 200

from paths import psp_model_location
from paths import figures_location

def psp_projection(elev=0,
                   azim=0,
                   roll=0,
                   show=False,
                   dpi=2000,
                   distinct_heat_shield=False):
    """
    A function for getting the projection area of PSP when seen 
    from an aribtrary angle.

    Parameters
    ----------
    elev : float, optional
        Elevation angle of the view in degrees. The default is 0.
    azim : float, optional
        Azimuthal angle of the view in degrees. The default is 0.
    roll : float, optional
        Roll angle of the view in degrees. The default is 0.
    show : bool, optional
        Whether to show the plot for visual confirmation. 
        The default is False.
    dpi : int, optional
        The resolution of the render. The default is 2000.

    Returns
    -------
    projection_area : float
        The area of PSP as seen from the angle on input.

    """

    # Create a new plot
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(projection='3d',computed_zorder=False)
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim, roll=roll)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file(psp_model_location)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors,
                                                       facecolors=r"#999999",
                                                       edgecolors="none",
                                                       antialiased=False))
    # Add a unit sphere.
    u = np.linspace(0, 2 * np.pi, 1000)
    v = np.linspace(0, np.pi, 1000)
    x = 5 + 1/np.sqrt(np.pi) * np.outer(np.cos(u), np.sin(v))
    y = -4 + 1/np.sqrt(np.pi) * np.outer(np.sin(u), np.sin(v))
    z = -4 + 1/np.sqrt(np.pi) * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=r"#000000", antialiased=False)

    #shield
    if distinct_heat_shield:
        if -90 < elev < 90 and 0 < azim < 180:
            mask = np.zeros(0,dtype=bool)
            for i in range(len(your_mesh.vectors[:,0,0])):
                face = your_mesh.vectors[i,:,:]
                include = np.any(face[:,1]>=1.480)
                mask = np.append(mask,include)
            shield = your_mesh.vectors[mask,:,:]
            ax.add_collection3d(mplot3d.art3d.Poly3DCollection(
                shield,
                facecolors=r"#cccccc",
                edgecolors="none",
                antialiased=False))

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    # Show the plot to the screen
    ax.set_aspect('equal')
    plt.axis('off')
    
    # Now we can save it to a numpy array.
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    if show:
        plt.show()
    #plt.close()

    # Count the PSP pixels.
    unique, counts = np.unique(data, return_counts=True)
    projection_area = counts[1]/counts[0]

    return projection_area


#%%
if __name__ == "__main__":
    print(psp_projection(20,60,0,1,distinct_heat_shield=True))