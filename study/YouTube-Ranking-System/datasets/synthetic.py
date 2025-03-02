import torch
import numpy as np
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    """
    YouTube 랭킹 시스템 테스트를 위한 합성 데이터셋
    """
    
    def __init__(self, num_samples=10000, num_categorical_fields=5, num_numerical_fields=3, 
                 categorical_field_dims=[100, 50, 30, 20, 10], num_tasks=2, seed=42):
        """
        합성 데이터셋 초기화
        
        :param num_samples: 생성할 샘플 수
        :param num_categorical_fields: 범주형 필드 수
        :param num_numerical_fields: 수치형 필드 수
        :param categorical_field_dims: 각 범주형 필드의 차원 (가능한 값의 수)
        :param num_tasks: 태스크 수
        :param seed: 랜덤 시드
        """
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_samples = num_samples
        self.num_categorical_fields = num_categorical_fields
        self.num_numerical_fields = num_numerical_fields
        self.field_dims = categorical_field_dims
        self.num_tasks = num_tasks
        
        # 범주형 피처 생성 (각 필드별로 범주 샘플링)
        self.categorical_features = torch.zeros(num_samples, num_categorical_fields, dtype=torch.long)
        for i in range(num_categorical_fields):
            self.categorical_features[:, i] = torch.randint(0, categorical_field_dims[i], (num_samples,))
        
        # 수치형 피처 생성 (0~1 사이의 난수)
        self.numerical_features = torch.rand(num_samples, num_numerical_fields)
        
        # 타겟 생성 (참여도와 만족도 등 여러 태스크를 위한 라벨)
        self.targets = torch.zeros(num_samples, num_tasks)
        
        # 간단한 모델로 라벨 생성 (피처와 라벨 간의 관계 부여)
        for i in range(num_samples):
            # 각 태스크마다 다른 로직으로 라벨 생성
            for task in range(num_tasks):
                # 범주형 피처에 따른 가중치 계산
                cat_weight = sum([
                    (self.categorical_features[i, j] / categorical_field_dims[j]) * (j+1)
                    for j in range(num_categorical_fields)
                ]) / num_categorical_fields
                
                # 수치형 피처에 따른 가중치 계산
                num_weight = self.numerical_features[i].mean().item()
                
                # 태스크별로 다른 피처 중요도 부여
                if task == 0:  # 첫 번째 태스크 (참여도)
                    p = 0.7 * cat_weight + 0.3 * num_weight
                else:  # 두 번째 태스크 (만족도)
                    p = 0.4 * cat_weight + 0.6 * num_weight
                
                # 노이즈 추가
                p = p + 0.2 * np.random.randn()
                p = max(0, min(1, p))  # 0~1 사이로 클립
                
                # 확률에 따라 이진 라벨 생성
                self.targets[i, task] = 1 if np.random.rand() < p else 0
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        return self.categorical_features[index], self.numerical_features[index], self.targets[index]
    
    def get_train_test_split(self, test_size=0.2):
        """
        학습 및 테스트 세트로 분할
        
        :param test_size: 테스트 세트 비율
        :return: 학습용 데이터셋, 테스트용 데이터셋
        """
        test_size = int(self.num_samples * test_size)
        train_indices = list(range(self.num_samples - test_size))
        test_indices = list(range(self.num_samples - test_size, self.num_samples))
        
        train_dataset = torch.utils.data.Subset(self, train_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        
        return train_dataset, test_dataset
    
    @property
    def numerical_num(self):
        """수치형 피처 수 반환"""
        return self.num_numerical_fields 