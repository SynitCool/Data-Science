import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, StratifiedKFold

def show_model_info(models):
    print("Model info\n--------------------")
    
    for name, model in models.items():
        print(f"Model name : {name}")
        
def show_scores(scores):
    for name, score in scores.items():
        print(f"{name} : {score}")
        
def find_mean_scores(scores):
    mean_scores = {}
    for name in scores.keys():
        model = name.split()[1]
        scr_type = name.split()[0]
        
        mean_score = np.mean(scores[name])
        
        mean_scores[f"{scr_type} mean {model} score"] = mean_score
        
    return mean_scores
    
class ModelSelection:
    def __init__(self, x, y, models):
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.models = {}
        
        try:
            self.len_models = len(models)
            
            for model in models:
                model_name = str(model).split('(')[0]
                
                self.models[model_name] = model
        except:
            print("Input model type with list or array")
            
    
    def use_train_test_split(self, random_state=42, shuffle=True, test_size=0.33):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_size, 
                                                            shuffle=shuffle, 
                                                            random_state=random_state)
        
        train_scores = {}
        test_scores = {}
        bias_scores = {}
        
        show_model_info(self.models)
        
        print("\nTraining models\n--------------------")
        start_train = time.time()
        for name in self.models.keys():
            self.models[name].fit(X_train, y_train)
            
            train_scores[f'Train {name} score'] = self.models[name].score(X_train, y_train)
            test_scores[f'Test {name} score'] = self.models[name].score(X_test, y_test)
            bias_scores[f'Bias {name} score'] = train_scores[f'Train {name} score'] - test_scores[f'Test {name} score']
            
            print(f"Done Training {name} in time:{time.time() - start_train}")

        print(f"--------------------\nDone Training models in time : {time.time() - start_train}\n")
        print("\nThe Scores :\n")
        
        show_scores(train_scores)
        print('-'*10)
        show_scores(test_scores)
        print('-'*10)
        show_scores(bias_scores)
        print('-'*10)
    
    def use_stratifiedkfold(self, nfolds=5, random_state=42, shuffle=True):
        skf = StratifiedKFold(nfolds, shuffle=shuffle, random_state=random_state)
        
        train_scores = {}
        test_scores = {}
        bias_scores = {}
        
        for fold, (trn_idx, val_idx) in enumerate(skf.split(self.x, self.y), 1):
            print(f"Fold {fold}\n----------")
            
            print(f"\nTraining models")
            start_train = time.time()
            for name in self.models.keys():
                self.models[name].fit(self.x[trn_idx], self.y[trn_idx])
                
                train_score = self.models[name].score(self.x[trn_idx], self.y[trn_idx])
                test_score = self.models[name].score(self.x[val_idx], self.y[val_idx])
                bias_score = train_score - test_score
                
                train_scores[f'Train {name} score'] = []
                test_scores[f'Test {name} score'] = []
                bias_scores[f'Bias {name} score'] = []
                
                train_scores[f'Train {name} score'].append(train_score)
                test_scores[f'Test {name} score'].append(test_score)
                bias_scores[f'Bias {name} score'].append(bias_score)
                
                print('\n--------------------')
                print(f"Done Training {name} in time:{time.time() - start_train}")
                
                print("\nThe Scores :\n")
                
                print(f"Train {name} score : {train_score}")
                print(f"Test {name} score : {test_score}")
                print(f"Bias {name} score : {bias_score}")
            
            print(f"--------------------\nDone Training models in time : {time.time() - start_train}\n")
        
        train_mean_scores = find_mean_scores(train_scores)
        test_mean_scores = find_mean_scores(test_scores)
        bias_mean_scores = find_mean_scores(bias_scores)
        
        print("\nThe Mean Scores :\n")
        
        show_scores(train_mean_scores)
        print('-'*10)
        show_scores(test_mean_scores)
        print('-'*10)
        show_scores(bias_mean_scores)
        print('-'*10)
    
    