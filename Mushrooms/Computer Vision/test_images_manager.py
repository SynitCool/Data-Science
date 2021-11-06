import os
import numpy as np
import pandas as pd

dataset_path = "Dataset\\Test Mushrooms"

labels = []
images = []
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    for image in os.listdir(label_path):
        image_path = os.path.join(label_path, image)
        
        images.append(image_path)
        labels.append(label)

images_df = pd.DataFrame(columns=["Image", "Label"])
images_df["Image"] = images
images_df["Label"] = labels

images_df.to_csv("Dataset\\test_images_mushrooms.csv", index=False)