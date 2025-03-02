import torch
from .layers import EmbeddingLayer, MultiLayerPerceptron, GatingNetwork, SoftmaxLayer

class SideTower(torch.nn.Module):
    """
    선택 편향을 보정하기 위한 Shallow 타워 (Wide 부분)
    참고: Wide & Deep 아키텍처에서 Wide 부분에 해당
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            input_dim, hidden_dims, dropout, 
            output_layer=True,
            activation='relu'
        )
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size,)``인 logit 텐서 
        """
        return self.mlp(x).squeeze(1)

class SharedBottomLayer(torch.nn.Module):
    """
    입력 특성을 처리하는 공유 은닉 계층
    논문에서 언급된 "공유 은닉 계층"을 구현
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.2):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            input_dim, hidden_dims, dropout,
            output_layer=False,
            activation='relu'
        )
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size, output_dim)``인 Float 텐서
        """
        return self.mlp(x)

class GatingNetworkLayer(torch.nn.Module):
    """
    Softmax 게이팅 네트워크 레이어
    """
    def __init__(self, input_dim, expert_num):
        super().__init__()
        self.gates = SoftmaxLayer(input_dim, expert_num)
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size, expert_num)``인 게이팅 가중치
        """
        return self.gates(x)

class ExpertLayer(torch.nn.Module):
    """
    전문가 레이어 구현
    """
    def __init__(self, input_dim, output_dim, dropout=0.2, activation='relu'):
        super().__init__()
        self.mlp = MultiLayerPerceptron(
            input_dim, [output_dim], dropout, 
            output_layer=False,
            activation=activation
        )
        
    def forward(self, x):
        """
        :param x: 크기가 ``(batch_size, input_dim)``인 Float 텐서
        :return: 크기가 ``(batch_size, output_dim)``인 Float 텐서
        """
        return self.mlp(x)

class YouTubeRankingModel(torch.nn.Module):
    """
    YouTube 비디오 추천 멀티태스크 랭킹 시스템 구현
    
    Reference:
        Zhe Zhao, et al. Recommending What Video to Watch Next: A Multitask Ranking System. RecSys 2019.
    """
    
    def __init__(
        self, 
        categorical_field_dims,       # 범주형 필드 차원 리스트
        numerical_num,                # 수치형 피처 개수
        embed_dim=64,                 # 임베딩 차원
        shared_bottom_dims=(512, 256),  # 공유 은닉 계층 차원
        shared_expert_num=4,          # 공유 전문가 수
        task_expert_num=2,            # 태스크별 전문가 수
        bottom_mlp_dims=(256, 128),   # 하단 MLP 차원
        tower_mlp_dims=(64, 32),      # 타워 MLP 차원
        side_tower_dims=(64, 32),     # 사이드 타워 차원
        task_num=2,                   # 태스크 수 (참여도, 만족도)
        dropout=0.2,
        use_wide_and_deep=True,       # Wide & Deep 아키텍처 사용 여부
    ):
        super().__init__()
        
        # 범주형 및 수치형 피처를 위한 임베딩 레이어
        self.embedding = EmbeddingLayer(categorical_field_dims, embed_dim)
        self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        
        # 임베딩 출력 차원 계산
        self.embed_output_dim = (len(categorical_field_dims) + 1) * embed_dim
        self.task_num = task_num
        self.use_wide_and_deep = use_wide_and_deep
        
        # 공유 은닉 계층 (입력 처리를 위한 첫 번째 계층)
        self.shared_bottom = SharedBottomLayer(self.embed_output_dim, shared_bottom_dims, dropout)
        
        # 공유 은닉 계층의 출력 차원이 전문가의 입력 차원
        self.shared_bottom_output_dim = self.shared_bottom.output_dim
        
        # 선택 편향 보정을 위한 사이드 타워 (Wide 부분)
        self.side_tower = SideTower(self.embed_output_dim, side_tower_dims, dropout)
        
        # 공유 전문가 (Mixture-of-Experts) (Deep 부분)
        self.shared_experts = torch.nn.ModuleList([
            ExpertLayer(self.shared_bottom_output_dim, bottom_mlp_dims[-1], dropout, 'relu')
            for _ in range(shared_expert_num)
        ])
        
        # 참여도 및 만족도 태스크별 전문가
        self.task_experts = torch.nn.ModuleList([
            torch.nn.ModuleList([
                ExpertLayer(self.shared_bottom_output_dim, bottom_mlp_dims[-1], dropout, 'relu')
                for _ in range(task_expert_num)
            ])
            for _ in range(task_num)
        ])
        
        # 공유 및 태스크별 전문가 수 기록
        self.shared_expert_num = shared_expert_num
        self.task_expert_num = task_expert_num
        self.total_expert_num = shared_expert_num + task_expert_num
        
        # 각 태스크별 게이팅 네트워크 
        self.gates = torch.nn.ModuleList([
            GatingNetworkLayer(self.shared_bottom_output_dim, self.total_expert_num)
            for _ in range(task_num)
        ])
        
        # 태스크별 타워 네트워크
        self.towers = torch.nn.ModuleList([
            MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout, activation='sigmoid')
            for _ in range(task_num)
        ])
        
        # 최종 랭킹 스코어 계산을 위한 가중치 조합 레이어
        self.ranking_weight = torch.nn.Parameter(torch.ones(task_num) / task_num)
        
    def forward(self, categorical_x, numerical_x):
        """
        :param categorical_x: 크기가 ``(batch_size, categorical_field_dims)``인 Long 텐서
        :param numerical_x: 크기가 ``(batch_size, numerical_num)``인 Float 텐서
        :return: 태스크별 예측 및 최종 랭킹 스코어
        """
        # 범주형 및 수치형 피처 임베딩
        categorical_emb = self.embedding(categorical_x)  # (batch_size, field_num, embed_dim)
        numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)  # (batch_size, 1, embed_dim)
        
        # 모든 임베딩을 하나의 벡터로 결합
        emb = torch.cat([categorical_emb, numerical_emb], 1).view(-1, self.embed_output_dim)
        
        # 선택 편향 보정 로짓 계산 (Wide 부분)
        selection_bias_logit = self.side_tower(emb)
        
        # 공유 은닉 계층을 통과 (첫 번째 특성 추출)
        shared_features = self.shared_bottom(emb)
        
        # 공유 전문가 출력 계산
        shared_expert_outputs = torch.stack([expert(shared_features) for expert in self.shared_experts], dim=1)
        # (batch_size, shared_expert_num, bottom_mlp_dims[-1])
        
        # 각 태스크별 예측값 계산
        task_outputs = []
        for i in range(self.task_num):
            # 태스크별 전문가 출력 계산
            task_expert_outputs = torch.stack([expert(shared_features) for expert in self.task_experts[i]], dim=1)
            # (batch_size, task_expert_num, bottom_mlp_dims[-1])
            
            # 전체 전문가 출력 결합 (공유 + 태스크별)
            expert_outputs = torch.cat([shared_expert_outputs, task_expert_outputs], dim=1)
            # (batch_size, total_expert_num, bottom_mlp_dims[-1])
            
            # 게이트 값 계산
            gate_values = self.gates[i](shared_features).unsqueeze(1)  # (batch_size, 1, total_expert_num)
            
            # 게이트 값과 전문가 출력 결합
            combined_features = torch.bmm(gate_values, expert_outputs).squeeze(1)  # (batch_size, bottom_mlp_dims[-1])
            
            # 타워 네트워크를 통과시켜 태스크별 예측값 계산
            task_output = torch.sigmoid(self.towers[i](combined_features).squeeze(1))  # (batch_size,)
            
            # Wide & Deep 아키텍처를 사용하는 경우
            if self.use_wide_and_deep and i == 0:  # 참여도 태스크에 대해서만 선택 편향 적용
                # 선택 편향(Wide 부분)과 사용자 유틸리티(Deep 부분) 결합
                bias_weight = torch.sigmoid(selection_bias_logit)
                task_output = task_output * (1 - bias_weight) + bias_weight * 0.5
            
            task_outputs.append(task_output)
        
        # 소프트맥스 가중치 계산
        normalized_weights = torch.nn.functional.softmax(self.ranking_weight, dim=0)
        
        # 최종 랭킹 스코어 계산 (가중 조합)
        ranking_score = torch.zeros_like(task_outputs[0])
        for i in range(self.task_num):
            ranking_score += normalized_weights[i] * task_outputs[i]
        
        # 태스크별 예측값, 선택 편향 보정 로짓, 최종 랭킹 스코어 반환
        return {
            'task_outputs': task_outputs,  # 태스크별 예측값 (참여도, 만족도)
            'selection_bias_logit': selection_bias_logit,  # 선택 편향 보정 로짓
            'ranking_score': ranking_score  # 최종 랭킹 스코어
        } 