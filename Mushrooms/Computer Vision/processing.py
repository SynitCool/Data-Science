import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Processing:
    def __init__(self, df):
        self.df = df
        
        self.indexes = []
        self.images = []
        self.labels = []
        
        self.pca = None
        self.pca_ratio = None
        
        self.ica = None
        
        self.lda = None
        
        for label in df["Label"].value_counts().index:
            for _ in range(25):
                min_index = np.min(list(df[df["Label"] == label].index))
                max_index = np.max(list(df[df["Label"] == label].index))
                
                self.indexes.append(np.random.randint(min_index, max_index))
                self.labels.append(label)
    
    def read_images(self):
        for index in self.indexes:
            image_path = self.df.iloc[index]["Image"]
            image = cv2.imread(image_path)
            self.images.append(image)
            
    def change_color(self, color=cv2.COLOR_BGR2RGB):
        mod_images = []
        for image in self.images:
            img = cv2.cvtColor(image, color)
            mod_images.append(img)
            
        self.images = mod_images
        
    def show_images(self, cmap='BrBG'):
        fig, ax = plt.subplots(figsize=(15, 9), nrows=2, ncols=2)
        
        rand_indexes = np.random.choice(len(self.images), 4)
        for i in range(4):
            if i < 2:
                ax[i, 0].imshow(self.images[rand_indexes[i]], cmap=cmap)
            
            else:
                i_ = i - 2 
                ax[i_, 1].imshow(self.images[rand_indexes[i]], cmap=cmap)
    
    def resize_images(self, size):
        mod_images = []
        
        for image in self.images:
            img = cv2.resize(image, size)
            mod_images.append(img)
            
        self.images = mod_images
    
    def filter_images(self, kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]), 
                      ddept=-1):
        mod_images = []
        for image in self.images:
            img = np.clip(cv2.filter2D(image, ddept, kernel), 0, 255)
            mod_images.append(img)
            
        self.images = mod_images
        
    def blur_images(self, blur=cv2.GaussianBlur, kernel=(3,3), sigma_x=3):
        mod_images = []
        if blur == cv2.medianBlur:
            for image in self.images:
                img = blur(image, kernel[0])
                mod_images.append(img)
        elif blur == cv2.blur:
            for image in self.images:
                img = blur(image, kernel)
                mod_images.append(img)
        else:
            for image in self.images:
                img = blur(image, kernel, sigma_x)
                mod_images.append(img)
    
        self.images = mod_images
    
    def threshold(self, threshold=cv2.threshold, thresh=125, max_value=255, 
                  threshold_type=cv2.THRESH_BINARY, 
                  adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                  block_size=13, c=3):
        
        mod_images = []
        if threshold == cv2.adaptiveThreshold:
            for image in self.images:
                img = threshold(image, max_value, adaptive_method, threshold_type,
                           block_size, c)
                mod_images.append(img)
        else:
            for image in self.images:
                thresh, img = threshold(image, thresh, max_value, threshold_type)
                mod_images.append(img)
        
        self.images = mod_images
        
    def adjust_contrast(self, contrast=1.0, brightness=0):
        mod_images = []
        for image in self.images:
            if len(image) == 3:
                image[:, :, :] = np.clip(contrast * image[:, :, :] + brightness,
                                         0, 255)
                mod_images.append(image)
            else:
                image[:, :] = np.clip(contrast * image[:, :] + brightness, 
                                      0, 255)
                mod_images.append(image)
                
        self.images = mod_images
        
    def edge(self, edge=cv2.Canny,threshold_1=125, threshold_2=255):
        mod_images = []
        if edge == cv2.Canny:
            for image in self.images:
                img = edge(image, threshold_1, threshold_2)
                mod_images.append(img)
                
        self.images = mod_images
        
    def contour(self, threshold_1=125, threshold_2=255):
        mod_images = []
        for image in self.images:
            canny = cv2.Canny(image, threshold_1, threshold_2)
            contour, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, 
                                                   cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.drawContours(image, contour, -1, 255, 0)
            mod_images.append(img)
            
        self.images = mod_images
        
    def cvtPCA(self):
        mod_images = []
        for image in self.images:
            mod_images.append(np.ravel(image))
            
        scaler = MinMaxScaler()
        img_scale = scaler.fit_transform(mod_images)
        
        pca = PCA(n_components=3)
        
        self.pca = pca.fit_transform(img_scale)
        self.pca_ratio = pca.explained_variance_ratio_
        
    def scatter_pca(self):
        if type(self.pca) != np.ndarray:
            print("Convert to PCA first")
            return 0
        
        labels_array = np.array(self.labels)
        pca_array = np.array(self.pca)
        
        fig = plt.figure(figsize=(15, 9))
        
        ax = fig.add_subplot(projection='3d')
        
        for label in np.unique(labels_array):
            indexes = []
            for i in range(labels_array.shape[0]):
                if labels_array[i] == label:
                    indexes.append(i)
                
                if len(indexes) == 25:
                    break
                    
            ax.scatter(pca_array[indexes, 0], pca_array[indexes, 1], pca_array[indexes, 2], label=label)
        
        ax.set_xlabel("First PCA")
        ax.set_ylabel("Second PCA")
        ax.set_zlabel("Third PCA")
        
        ax.legend()
        
    def cvtICA(self):
        mod_images = []
        for image in self.images:
            mod_images.append(np.ravel(image))
            
        scaler = MinMaxScaler()
        img_scale = scaler.fit_transform(mod_images)
        
        ica = FastICA(n_components=3)            
                
        self.ica = ica.fit_transform(img_scale)
        
    def scatter_ica(self):
        if type(self.ica) != np.ndarray:
            print("Convert to ICA first")
            return 0
        
        labels_array = np.array(self.labels)
        ica_array = np.array(self.ica)
        
        fig = plt.figure(figsize=(15, 9))
        
        ax = fig.add_subplot(projection='3d')
        
        for label in np.unique(labels_array):
            indexes = []
            for i in range(labels_array.shape[0]):
                if labels_array[i] == label:
                    indexes.append(i)
                
                if len(indexes) == 25:
                    break
                    
            ax.scatter(ica_array[indexes, 0], ica_array[indexes, 1], ica_array[indexes, 2], label=label)
        
        ax.set_xlabel("First ICA")
        ax.set_ylabel("Second ICA")
        ax.set_zlabel("Third ICA")
        
        ax.legend()
        
    def cvtLDA(self):
        mod_images = []
        for image in self.images:
            mod_images.append(np.ravel(image))
            
        scaler = MinMaxScaler()
        le = LabelEncoder()
        lda = LinearDiscriminantAnalysis()
        
        lbl_encode = le.fit_transform(self.labels)
        img_scale = scaler.fit_transform(mod_images)
        lda = lda.fit_transform(img_scale, lbl_encode)
        
        self.lda = lda
        
    def scatter_lda(self):
        if type(self.lda) != np.ndarray:
            print("Convert to LDA first")
            return 0
        
        labels_array = np.array(self.labels)
        lda_array = np.array(self.lda)
        
        fig = plt.figure(figsize=(15, 9))
        
        ax = fig.add_subplot(projection='3d')
        
        for label in np.unique(labels_array):
            indexes = []
            for i in range(labels_array.shape[0]):
                if labels_array[i] == label:
                    indexes.append(i)
                
                if len(indexes) == 25:
                    break
                    
            ax.scatter(lda_array[indexes, 0], lda_array[indexes, 1], lda_array[indexes, 2], label=label)
        
        ax.set_xlabel("First LDA")
        ax.set_ylabel("Second LDA")
        ax.set_zlabel("Third LDA")
        
        ax.legend()
                
                
                
                
                
                
                
                
                
                