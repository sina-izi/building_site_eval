import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from perlin_noise import PerlinNoise
from itertools import product, combinations
from scipy.ndimage import gaussian_filter

# Define the grid for the surface
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)

# Create Perlin noise generator
noise = PerlinNoise(octaves=4, seed=1)

# Define the elevation function using Perlin noise
original_z = np.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        original_z[i, j] = noise([x[i, j] * 0.1, y[i, j] * 0.1]) * 0.5

# Copy of the original elevation data
z = np.copy(original_z)

# Create the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with transparency
surface_plot = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.7)

# Add labels and title
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')
ax.set_title('Realistic Earth Surface')

# Create a movable cube
cube_size = 0.5
cube_center = [0, 0, 0.5]
cube = [cube_center[0] - cube_size / 2, cube_center[0] + cube_size / 2,
        cube_center[1] - cube_size / 2, cube_center[1] + cube_size / 2,
        cube_center[2] - cube_size / 2, cube_center[2] + cube_size / 2]

# Track the previous cube center
previous_cube_center = cube_center.copy()

# Draw the initial cube
cube_artists = []
def draw_cube():
    global cube_artists
    for artist in cube_artists:
        artist[0].remove()
    cube_artists = []
    for s, e in combinations(np.array(list(product(cube[0:2], cube[2:4], cube[4:6]))), 2):
        if np.sum(np.abs(s - e)) == cube_size:
            cube_artists.append(ax.plot3D(*zip(s, e), color="red"))
    draw_intersection_plane()
    plt.draw()

# Define a function to draw the intersection plane
def draw_intersection_plane():
    global x, y, z, cube_center, cube_size
    cx, cy, cz = cube_center
    mask = (x >= cx - cube_size / 2) & (x <= cx + cube_size / 2) & \
           (y >= cy - cube_size / 2) & (y <= cy + cube_size / 2)
    intersection_x = x[mask]
    intersection_y = y[mask]
    intersection_z = z[mask]
    if len(intersection_x) >= 3 and len(intersection_y) >= 3 and len(intersection_z) >= 3:
        ax.plot_trisurf(intersection_x, intersection_y, intersection_z, color='k', alpha=0.5)

draw_cube()

# Define the zoom function
def zoom(event):
    scale_factor = 1.1
    if event.button == 'up':
        ax.set_xlim(ax.get_xlim3d()[0] * scale_factor, ax.get_xlim3d()[1] * scale_factor)
        ax.set_ylim(ax.get_ylim3d()[0] * scale_factor, ax.get_ylim3d()[1] * scale_factor)
        ax.set_zlim(ax.get_zlim3d()[0] * scale_factor, ax.get_zlim3d()[1] * scale_factor)
    elif event.button == 'down':
        ax.set_xlim(ax.get_xlim3d()[0] / scale_factor, ax.get_xlim3d()[1] / scale_factor)
        ax.set_ylim(ax.get_ylim3d()[0] / scale_factor, ax.get_ylim3d()[1] / scale_factor)
        ax.set_zlim(ax.get_zlim3d()[0] / scale_factor, ax.get_zlim3d()[1] / scale_factor)
    plt.draw()

# Function to reset the previously smoothed area
def reset_previous_area():
    global z
    prev_cx, prev_cy = previous_cube_center[0], previous_cube_center[1]
    dist = np.sqrt((x - prev_cx) ** 2 + (y - prev_cy) ** 2)
    influence_radius = cube_size * 8  # Increased influence radius

    # Reset the elevation within the influence radius to original values
    mask = dist < influence_radius
    z[mask] = original_z[mask]

# Function to smoothen the surface around the cube
def smoothen_surface():
    global z, surface_plot, previous_cube_center
    reset_previous_area()

    cx, cy = cube_center[0], cube_center[1]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    influence_radius = cube_size * 6  # Influence radius

    # Apply Gaussian smoothing within the influence radius
    mask = dist < influence_radius
    z[mask] = gaussian_filter(z, sigma=8)[mask]  # Increase sigma for stronger smoothing
    # Redraw the surface
    for artist in ax.collections:
        artist.remove()
    surface_plot = ax.plot_surface(x, y, z, cmap='terrain', alpha=0.7)
    draw_cube()
    plt.draw()

# Define the function to move the cube with keys
def on_key(event):
    global cube_center, previous_cube_center
    previous_cube_center = cube_center.copy()
    if event.key == 'w':
        cube_center[1] += 0.1
    elif event.key == 'x':
        cube_center[1] -= 0.1
    elif event.key == 'a':
        cube_center[0] -= 0.1
    elif event.key == 'd':
        cube_center[0] += 0.1
    elif event.key == 'e':
        cube_center[2] += 0.1
    elif event.key == 'r':
        cube_center[2] -= 0.1
    cube[:] = [cube_center[0] - cube_size / 2, cube_center[0] + cube_size / 2,
               cube_center[1] - cube_size / 2, cube_center[1] + cube_size / 2,
               cube_center[2] - cube_size / 2, cube_center[2] + cube_size / 2]
    smoothen_surface()

# Connect the events
fig.canvas.mpl_connect('scroll_event', zoom)
fig.canvas.mpl_connect('key_press_event', on_key)

# Show the plot
plt.show()

