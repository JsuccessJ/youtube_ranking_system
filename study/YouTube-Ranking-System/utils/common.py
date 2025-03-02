import torch
import numpy as np
import logging
import os
from sklearn.metrics import roc_auc_score

def setup_logger(name, log_file, level=logging.INFO):
    """로깅 설정을 위한 함수"""
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # 콘솔에도 출력
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    return logger

class EarlyStopper:
    """조기 종료를 위한 클래스"""
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        
    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            # 모델 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def evaluate_metrics(y_true, y_pred):
    """성능 평가 지표 계산"""
    auc = roc_auc_score(y_true, y_pred)
    loss = torch.nn.functional.binary_cross_entropy(
        torch.tensor(y_pred, dtype=torch.float), 
        torch.tensor(y_true, dtype=torch.float), 
        reduction='mean'
    ).item()
    return auc, loss 