import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize

import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y, y_true, y_pred, model, ax=None):
    le = LabelEncoder()
    le.fit(y)
    
    labels = le.inverse_transform(np.unique(y_true))
    c_matrix = confusion_matrix(y_true, y_pred)
    
    confusion_matrix_df = pd.DataFrame(c_matrix, index=labels, columns=labels)
    
    if ax == None:
        sns.heatmap(confusion_matrix_df, annot=True)
        
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title(f"{model} Confusion Matrix")
        
    elif ax != None:
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
        
        ax.set_title(f"{model} Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
    
def calc_roc_curve(y_test, y_proba):
    # Initialize variable
    y = label_binarize(y_test, classes=np.unique(y_test))
    
    n_classes = y.shape[1]

    fprs = {}
    tprs = {}
    aucs = {}
    
    # Make true positive, false positive, area under curve score of each class
    for i in range(n_classes):
        fprs[i+1], tprs[i+1], _ = roc_curve(y[:, i], y_proba[:, i])
        
        aucs[i+1] = auc(fprs[i+1], tprs[i+1])
        
    # Make micro score
    fprs['micro'], tprs['micro'], _ = roc_curve(y.ravel(), y_proba.ravel())
    aucs['micro'] = auc(fprs['micro'], tprs['micro'])
    
    # Make macro score
    all_fprs = np.unique(np.concatenate([fprs[i+1] for i in range(n_classes)]))
    
    mean_tprs = np.zeros_like(all_fprs)
    for i in range(n_classes):
        mean_tprs += np.interp(all_fprs, fprs[i+1], tprs[i+1])
    
    mean_tprs /= n_classes
    
    fprs['macro'] = all_fprs
    tprs['macro'] = mean_tprs
    aucs['macro'] = auc(fprs['macro'], tprs['macro'])
    
    return fprs, tprs, aucs

def plot_roc_curve(fprs, tprs, aucs, model, ax=None):
    if ax == None:
        for fpr_name, tpr_name, auc_name in zip(fprs.keys(), tprs.keys(), aucs.keys()):
            auc = f"{model} Prediction {auc_name} (AUROC = {round(aucs[auc_name], 3)})"
            plt.plot(fprs[fpr_name], tprs[tpr_name], marker='.', label=auc)
            
            plt.legend()
            
        plt.title(f"{model} ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        
    elif ax != None:
        for fpr_name, tpr_name, auc_name in zip(fprs.keys(), tprs.keys(), aucs.keys()):
            auc = f"{model} Prediction {auc_name} (AUROC = {round(aucs[auc_name], 3)})"
            ax.plot(fprs[fpr_name], tprs[tpr_name], marker='.', label=auc)
            
            ax.legend()
            
        ax.set_title(f"{model} ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")










    