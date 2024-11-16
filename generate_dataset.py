import cv2
import numpy as np
import pandas as pd
from tqdm.contrib import tenumerate
import os

def gen_points(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to binary
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(binary, 50, 150)

    # Find contours of the line
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Choose the longest contour (most likely the red line)
    contour = max(contours, key=cv2.contourArea)

    # Specify the spacing for the points along the line
    point_spacing = 10  # Adjust this to control the distance between points

    # Sample points along the contour at regular intervals
    points = []
    for i in range(0, len(contour), point_spacing):
        x, y = contour[i][0]
        points.append((x, y))

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    img = img / 255.0
    
    return np.array(points) / 500.0 , img

def main(img_path, save_path):
    # Generate points for each image
    images = []
    all_points = []
    
    for i, file in tenumerate(os.listdir(img_path)):
        if file.endswith(".jpg"):
            image_path = os.path.join(img_path, file)
            points, image = gen_points(image_path)
            images.append(image)            
            ids = np.full((len(points), 1), i)
            points_ = np.hstack((ids, points))
            all_points.append(points_)
    
    all_points = np.vstack(all_points) 
    np.save(os.path.join(save_path, "imgs.npy"), images) 
    np.save(os.path.join(save_path, "pts.npy"), all_points)
            
if __name__ == "__main__":
    image_path = "datasets/line_follow/imgs/"
    save_path = "datasets/line_follow"
    main(image_path, save_path)