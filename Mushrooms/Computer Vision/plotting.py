import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def show_distribution(images):
    pixels = np.ravel(images)
    
    fig, ax = plt.subplots(figsize=(15, 9))
    
    sns.distplot(x=pixels)
    
    plt.show()