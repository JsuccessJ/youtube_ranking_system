import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron, GatingNetwork

class MMoEModel(torch.nn.Module):
    """
    Multi-gate Mixture-of-Experts 모델 구현
    
    Reference:
        Ma, Jiaqi, et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts. KDD 2018.
    """
    
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, 
                 tower_mlp_dims, task_num, expert_num, dropout=0.2):
        super().__init__()
        
        # 범주형 피처와 수치형 피처를 위한 임베딩 레이어
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        
        # 임베딩 출력 차원 계산
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num
        
        # Expert 네트워크 생성 (shared bottom)
        self.experts = torch.nn.ModuleList([
            MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) 
            for _ in range(expert_num)
        ])
        
        # 각 태스크별 게이트 네트워크
        self.gates = torch.nn.ModuleList([
            GatingNetwork(self.embed_output_dim, expert_num) for _ in range(task_num)
        ])
        
        # 각 태스크별 타워 네트워크 
        self.towers = torch.nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) 
            for _ in range(task_num)
        ])
        
    def forward(self, categorical_x, numerical_x):
        """
        :param categorical_x: 크기가 ``(batch_size, categorical_field_dims)``인 Long 텐서
        :param numerical_x: 크기가 ``(batch_size, numerical_num)``인 Float 텐서
        :return: 각 태스크별 예측값 리스트
        """
        # 범주형 및 수치형 피처 임베딩
        categorical_emb = self.embedding(categorical_x)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        
        # 모든 임베딩을 하나의 벡터로 결합
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        
        # 각 태스크별 게이트 값 계산
        gate_values = [gate(emb).unsqueeze(1) for gate in self.gates]  # shape: [(batch_size, 1, expert_num)]
        
        # 모든 전문가의 출력 계산
        expert_outputs = torch.cat([expert(emb).unsqueeze(1) for expert in self.experts], dim=1)  
        # expert_outputs shape: (batch_size, expert_num, bottom_mlp_dims[-1])
        
        # 각 태스크별로 게이트 값과 전문가 출력을 결합
        task_features = [
            torch.bmm(gate_values[i], expert_outputs).squeeze(1) for i in range(self.task_num)
        ]  # task_features shape: [(batch_size, bottom_mlp_dims[-1])]
        
        # 각 태스크별 타워 네트워크를 통과시켜 최종 예측값 계산
        results = [
            torch.sigmoid(self.towers[i](task_features[i]).squeeze(1)) for i in range(self.task_num)
        ]  # results shape: [(batch_size,)]
        
        return results 