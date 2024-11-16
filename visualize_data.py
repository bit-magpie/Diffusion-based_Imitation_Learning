import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import pandas as pd
from tqdm.contrib import tenumerate
import os
import pickle

def gen_points(image_path):
    # Load the image
    image = cv2.imread(image_path)

# Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour = max(contours, key=cv2.contourArea)

    point_spacing = 10  # Adjust this to control the distance between points

    # Sample points along the contour at regular intervals
    points = []
    
    for i in range(0, len(contour), point_spacing):
        x, y = contour[i][0]      
        points.append((x, y))

    out_image = np.ones_like(image) * 255

    out_image = cv2.drawContours(out_image, contours, -1, (73, 136, 200), 30)  # Draw in black color with thickness 1
    # out_image = cv2.circle(out_image, (x,y), radius=5, color=(255, 0, 0), thickness=20)

    
    # start = points[0]
    # points_ = np.array(points)
    # actions = np.insert(np.diff(points_, axis=0), 0, [0.0, 0.0], axis=0)
    
    return points, out_image


def save_animation(points, image):
    # Initial position
    initial_position = points[0]

    # Calculate cumulative positions
    positions = points

    # Setup the plot
    fig, ax = plt.subplots()
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[0])
    dot, = ax.plot([], [], 'ro', markersize=10)  # 'bo' represents a blue dot
    # ax.plot(pt[:,0], pt[:,1], linewidth=12, zorder=0)
    ax.imshow(image)
    ax.set_ylim(ax.get_ylim()[::-1])

    # Initialization function to set up the dot at the starting position
    def init():
        dot.set_data([initial_position[0]], [initial_position[1]])
        return dot,

    # Animation update function to move the dot
    def update(frame):
        x, y = positions[frame]
        dot.set_data([x], [y])  # Pass lists or arrays to set_data
        return dot,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True)

    # Save the animation as a GIF
    ani.save("dot_movement.gif", writer=PillowWriter(fps=15))

    print("Animation saved as 'dot_movement.gif'")

def main(image_path, save_path):

    points, image = gen_points(image_path)
    # plt.imshow(image)
    # plt.show()
    save_animation(points, image)

if __name__ == "__main__":
    image_path = "datasets/line_follow/imgs/line12.jpg"
    save_path = "datasets/line_follow"
    main(image_path, save_path)