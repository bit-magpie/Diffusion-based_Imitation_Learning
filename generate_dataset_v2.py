import cv2
import numpy as np
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
    images = []
    
    for i in range(0, len(contour), point_spacing):
        x, y = contour[i][0]      

        out_image = np.ones_like(image) * 255

        cv2.drawContours(out_image, contours, -1, (73, 136, 200), 30)  # Draw in black color with thickness 1
        out_image = cv2.circle(out_image, (x,y), radius=5, color=(255, 0, 0), thickness=20)
        out_image = cv2.resize(out_image, (32, 32), interpolation=cv2.INTER_AREA)
        images.append(out_image / 255.0)
        points.append((x / image.shape[0], y / image.shape[1]))
    
    start = points[0]
    points_ = np.array(points)
    actions = np.insert(np.diff(points_, axis=0), 0, [0.0, 0.0], axis=0)
    
    return actions, np.array(images), start

def main(img_path, save_path):
    # Generate points for each image
    all_images = []
    all_actions = []
    starts = []
    ids = []
    
    for i, file in tenumerate(os.listdir(img_path)):
        if file.endswith(".jpg"):
            image_path = os.path.join(img_path, file)
            actions, images, start = gen_points(image_path)
            starts.append(start)
            # print(actions.shape, images.shape)
            all_images.append(images)            
            all_actions.append(actions)
            
            ids.append(len(actions))
    
    np.save(os.path.join(save_path, "images.npy"), np.vstack(all_images)) 
    np.save(os.path.join(save_path, "actions.npy"), np.vstack(all_actions))
    pickle.dump([ids, starts], open(os.path.join(save_path, "starts.pkl"), 'wb'))



if __name__ == "__main__":
    image_path = "datasets/line_follow/imgs/"
    save_path = "datasets/line_follow"
    main(image_path, save_path)