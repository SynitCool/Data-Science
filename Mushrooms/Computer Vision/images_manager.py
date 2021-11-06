import pandas as pd
import os 

DATASET_PATH = "Dataset"
MUSHROOMS_PATH = os.path.join(DATASET_PATH, "Mushrooms")

labels = [label for label in os.listdir(MUSHROOMS_PATH)]

images_labels = []
for label in labels:
    label_path = os.path.join(MUSHROOMS_PATH, label)
    for image in os.listdir(label_path):
        image_path = os.path.join(label_path, image)
        image_label = [image_path, label]
        images_labels.append(image_label)
        
images_df = pd.DataFrame(images_labels, columns=["Image", "Label"])

df_path = os.path.join(DATASET_PATH, "images_mushrooms.csv")

images_df.to_csv(df_path, index=False)