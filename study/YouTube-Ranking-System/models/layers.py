import torch
import numpy as np

class EmbeddingLayer(torch.nn.Module):
    """특성 필드를 임베딩하는 레이어"""
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        # 모든 필드의 임베딩을 위한 단일 임베딩 테이블
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        # 각 필드의 오프셋 계산
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # 임베딩 가중치 초기화
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, num_fields)``인 Long 텐서
        :return: 크기가 ``(batch_size, num_fields, embed_dim)``인 임베딩 텐서
        """
        # 각 필드에 오프셋 적용
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):
    """다층 퍼셉트론 구현"""
    def __init__(self, input_dim, layer_dims, dropout=0.2, output_layer=True, activation='relu'):
        super().__init__()
        layers = list()
        
        # 히든 레이어 구성
        for embed_dim in layer_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            
            # 활성화 함수 선택
            if activation.lower() == 'relu':
                layers.append(torch.nn.ReLU())
            elif activation.lower() == 'sigmoid':
                layers.append(torch.nn.Sigmoid())
            elif activation.lower() == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
                
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        
        # 출력 레이어 추가 (선택적)
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: mlp 출력
        """
        return self.mlp(x)

class SoftmaxLayer(torch.nn.Module):
    """Softmax 게이팅 레이어"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size, output_dim)``인 확률 텐서
        """
        return self.softmax(self.linear(x))

class GatingNetwork(torch.nn.Module):
    """Mixture-of-Experts를 위한 게이팅 네트워크"""
    def __init__(self, input_dim, expert_num):
        super().__init__()
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(input_dim, expert_num),
            torch.nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size, expert_num)``인 게이팅 가중치
        """
        return self.gate(x) 