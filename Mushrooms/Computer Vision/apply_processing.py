import pandas as pd
import numpy as np
import cv2
import functools

def apply(functions_image, functions_label, df):
    mod_images = []
    mod_labels = []
    
    try:
        length_functions = len(functions_image)
        length_labels = len(functions_label)
    except:
        print("make list type functions_images and functions label")
    
    for func_image, func_label in zip(functions_image, functions_label):
        images = func_image(df)
        labels = func_label(df['Label'])
        
        mod_images.extend(images)
        mod_labels.extend(labels)
        
    return mod_images, mod_labels

def function_hsv(images):
    color = cv2.COLOR_BGR2HSV
    
    mod_images = list(map(cv2.imread, images))
    mod_images = map(functools.partial(cv2.cvtColor, code=color), mod_images)
    
    return list(mod_images)

def function_gray(images):
    color = cv2.COLOR_BGR2GRAY
    
    mod_images = list(map(cv2.imread, images))
    mod_images = map(functools.partial(cv2.cvtColor, code=color), mod_images)
    
    return list(mod_images)

def function_rgb(images):
    color = cv2.COLOR_BGR2RGB
    
    mod_images = list(map(cv2.imread, images))
    mod_images = map(functools.partial(cv2.cvtColor, code=color), mod_images)
    
    return list(mod_images)

def function_hls(images):
    color = cv2.COLOR_BGR2HLS
    
    mod_images = list(map(cv2.imread, images))
    mod_images = map(functools.partial(cv2.cvtColor, code=color), mod_images)
    
    return list(mod_images)

def function_label(labels):
    return labels

def blur_images(images, blur=cv2.GaussianBlur, kernel=(3,3), sigma_x=3):
    if blur == cv2.medianBlur:
        ksize = kernel[0]
        
        mod_images = list(map(functools.partial(cv2.medianBlur, ksize=ksize), images))
    
    elif blur == cv2.blur:
        mod_images = list(map(functools.partial(cv2.blur, ksize=kernel), images))
    
    else:
        mod_images = list(map(functools.partial(cv2.GaussianBlur, ksize=kernel, sigmaX=sigma_x), images))
    
    return mod_images

def adjust_contrast_images(images, contrast=1, brightness=0):
    mod_images = []
    for image in images:
        if len(image) == 3:
            image[:, :, :] = np.clip(contrast * image[:, :, :] + brightness, 0, 255)
            
            mod_images.append(image)
        else:
            image[:, :] = np.clip(contrast * image[:, :] + brightness, 0, 255)
            mod_images.append(image)
            
    return mod_images

def filter_images(images, kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]), ddepth=-1):
    mod_images = []
    for image in images:
        img = np.clip(cv2.filter2D(image, ddepth, kernel), 0, 255)
            
        mod_images.append(img)
    
    return mod_images

def edge_images(images, edge=cv2.Canny, threshold1=125, threshold2=255, ddepth=-1, dx=0, dy=0):
    if edge == cv2.Canny:
        mod_images = list(map(functools.partial(cv2.Canny, threshold1=threshold1, threshold2=threshold2), images))
            
    elif edge == cv2.Laplacian:
        mod_images = list(map(functools.partial(cv2.Laplacian, ddepth=ddepth), images))
        
    elif edge == cv2.Sobel:
        mod_images = list(map(functools.partial(cv2.Sobel, ddepth=ddepth, dx=dx, dy=dy), images))
        
    return mod_images

def contour_images(edge_images, drawned_images, mode=cv2.RETR_LIST, 
                   method=cv2.CHAIN_APPROX_SIMPLE, color=255, thickness=0, contouridx=-1):
    
    if len(edge_images) != len(drawned_images):
        print("Length of edge images and drawned_images must be same")
        return 'error: Length of edge images and drawned_images must be same'
    
    else:
        mod_images = []
        for drawned, edge in zip(drawned_images, edge_images):
            contour, hierarchies = cv2.findContours(edge, mode, method)
            
            img = cv2.drawContours(drawned, contour, contouridx, color, thickness)
            mod_images.append(img)
            
        return mod_images
    
def resize_images(images, size):
    mod_images = map(functools.partial(cv2.resize, dsize=size), images)
    
    return list(mod_images)
    
def rotate_images(images):
    rotates_code = [cv2.cv2.ROTATE_180, cv2.cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    flipped_code = [0]
    
    mod_images = []
    
    for roco, flipcode in zip(rotates_code, flipped_code):
        rotated_images = map(functools.partial(cv2.rotate, rotateCode=roco), images)
        flipped_images = map(functools.partial(cv2.flip, flipCode=flipcode), images)
        
        mod_images.extend(list(rotated_images))
        mod_images.extend(list(flipped_images))
        
    return mod_images

def threshold_images(images, threshold=cv2.threshold, thresh=125, max_value=255, 
                  threshold_type=cv2.THRESH_BINARY, 
                  adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                  block_size=13, c=3):
    
    if threshold == cv2.adaptiveThreshold:
        mod_images = map(functools.partial(cv2.adaptiveThreshold, maxValue=max_value, adaptiveMethod=adaptive_method, thresholdType=threshold_type,blockSize=block_size, C=c), images)
    else:
        thresh_img = map(functools.partial(cv2.threshold, thresh=thresh, maxval=max_value, type=threshold_type), images)
    
        thresh_images = list(thresh_img)
    
        thresh = np.array(thresh_images)[:, 0]
        mod_images = np.array(thresh_images)[:, 1]
    
    return mod_images
    
def rotated_labels(labels):
    mod_images = []
    for _ in range(4):
        mod_images.extend(labels)
        
    return mod_images
    
def kmeans_images(images, K=12):
    mod_images = []
    for image in images:
        Z = np.reshape(image, (-1, 3))
        Z = np.float32(Z)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = K

        ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((image.shape))
        
        mod_images.append(res2)
        
    return mod_images
    
    
    
    
    
    
    
    
    
    
    
                          
                          
                          
                          
                   