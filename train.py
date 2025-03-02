import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from datetime import datetime

from datasets import SyntheticDataset
from models import MMoEModel, YouTubeRankingModel
from utils.common import EarlyStopper, setup_logger, evaluate_metrics

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    """모델 학습 함수"""
    model.train()
    total_loss = 0
    loader = tqdm(data_loader, smoothing=0, mininterval=1.0)
    
    for i, (categorical_fields, numerical_fields, labels) in enumerate(loader):
        categorical_fields = categorical_fields.to(device)
        numerical_fields = numerical_fields.to(device)
        labels = labels.to(device)
        
        # 모델 전방 전파
        model_output = model(categorical_fields, numerical_fields)
        
        # YouTube 랭킹 모델과 MMoE 모델의 출력 형태 차이 처리
        if isinstance(model_output, dict):  # YouTube 랭킹 모델
            task_outputs = model_output['task_outputs']
            selection_bias_logit = model_output['selection_bias_logit']
            
            # 각 태스크별 손실 계산
            task_losses = [criterion(task_outputs[j], labels[:, j].float()) for j in range(labels.size(1))]
            
            # 선택 편향 손실 (가중치를 부여하여 중요도 조절 가능)
            selection_bias_loss = criterion(torch.sigmoid(selection_bias_logit), torch.ones_like(selection_bias_logit) * 0.5)
            
            # 전체 손실 계산
            loss = sum(task_losses) / len(task_losses) + 0.1 * selection_bias_loss
        else:  # MMoE 모델
            task_outputs = model_output
            task_losses = [criterion(task_outputs[j], labels[:, j].float()) for j in range(labels.size(1))]
            loss = sum(task_losses) / len(task_losses)
        
        # 역전파 및 최적화
        model.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            loader.set_postfix(loss=total_loss / log_interval)
            total_loss = 0

def test(model, data_loader, task_num, device):
    """모델 평가 함수"""
    model.eval()
    task_labels = [[] for _ in range(task_num)]
    task_predictions = [[] for _ in range(task_num)]
    
    with torch.no_grad():
        for categorical_fields, numerical_fields, labels in tqdm(data_loader, smoothing=0, mininterval=1.0):
            categorical_fields = categorical_fields.to(device)
            numerical_fields = numerical_fields.to(device)
            labels = labels.to(device)
            
            # 모델 전방 전파
            model_output = model(categorical_fields, numerical_fields)
            
            # YouTube 랭킹 모델과 MMoE 모델의 출력 형태 차이 처리
            if isinstance(model_output, dict):  # YouTube 랭킹 모델
                task_outputs = model_output['task_outputs']
            else:  # MMoE 모델
                task_outputs = model_output
            
            # 각 태스크별 실제값과 예측값 저장
            for i in range(task_num):
                task_labels[i].extend(labels[:, i].cpu().numpy())
                task_predictions[i].extend(task_outputs[i].cpu().numpy())
    
    # 성능 평가
    results = []
    for i in range(task_num):
        auc, loss = evaluate_metrics(task_labels[i], task_predictions[i])
        results.append((auc, loss))
    
    return results

def main(args):
    # 로그 디렉토리 생성
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 로거 설정
    logger = setup_logger(
        'youtube_ranking', 
        f"{args.log_dir}/train_{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logger.info(f"Arguments: {args}")
    
    # 디바이스 설정
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 데이터셋 생성
    logger.info("Creating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=args.num_samples,
        num_categorical_fields=args.num_categorical_fields,
        num_numerical_fields=args.num_numerical_fields,
        categorical_field_dims=[100, 50, 30, 20, 10],
        num_tasks=args.task_num,
        seed=args.seed
    )
    
    # 학습/테스트 분할
    train_dataset, test_dataset = dataset.get_train_test_split(test_size=args.test_size)
    
    # 데이터 로더 생성
    train_data_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # 모델 생성
    field_dims = dataset.field_dims
    numerical_num = dataset.numerical_num
    
    if args.model_name == 'mmoe':
        logger.info("Creating MMoE model...")
        model = MMoEModel(
            categorical_field_dims=field_dims,
            numerical_num=numerical_num,
            embed_dim=args.embed_dim,
            bottom_mlp_dims=args.bottom_mlp_dims,
            tower_mlp_dims=args.tower_mlp_dims,
            task_num=args.task_num,
            expert_num=args.expert_num,
            dropout=args.dropout
        ).to(device)
    elif args.model_name == 'youtube_ranking':
        logger.info("Creating YouTube Ranking model...")
        model = YouTubeRankingModel(
            categorical_field_dims=field_dims,
            numerical_num=numerical_num,
            embed_dim=args.embed_dim,
            shared_bottom_dims=args.shared_bottom_dims,
            shared_expert_num=args.shared_expert_num,
            task_expert_num=args.task_expert_num,
            bottom_mlp_dims=args.bottom_mlp_dims,
            tower_mlp_dims=args.tower_mlp_dims,
            side_tower_dims=args.side_tower_dims,
            task_num=args.task_num,
            dropout=args.dropout,
            use_wide_and_deep=args.use_wide_and_deep
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")
    
    # 손실 함수와 옵티마이저 정의
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 저장 경로 설정
    save_path = os.path.join(args.save_dir, f"{args.model_name}.pt")
    early_stopper = EarlyStopper(num_trials=args.patience, save_path=save_path)
    
    # 학습 수행
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        train(model, optimizer, train_data_loader, criterion, device)
        
        # 평가 수행
        results = test(model, test_data_loader, args.task_num, device)
        
        # 결과 출력
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        avg_auc = 0
        for i, (auc, loss) in enumerate(results):
            task_name = "Engagement" if i == 0 else "Satisfaction"
            logger.info(f"  Task {i+1} ({task_name}) - AUC: {auc:.4f}, Loss: {loss:.4f}")
            avg_auc += auc
        avg_auc /= len(results)
        
        # 조기 종료 체크
        if not early_stopper.is_continuable(model, avg_auc):
            logger.info(f"Early stopping at epoch {epoch+1}. Best average AUC: {early_stopper.best_accuracy:.4f}")
            break
    
    # 학습 완료
    logger.info("Training completed")
    
    # 최적 모델 로드
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        logger.info(f"Loaded best model from {save_path}")
    
    # 최종 테스트 수행
    logger.info("Final evaluation on test set")
    results = test(model, test_data_loader, args.task_num, device)
    
    # 최종 결과 출력
    for i, (auc, loss) in enumerate(results):
        task_name = "Engagement" if i == 0 else "Satisfaction"
        logger.info(f"  Task {i+1} ({task_name}) - AUC: {auc:.4f}, Loss: {loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YouTube Multitask Ranking System')
    
    # 데이터 관련 인자
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples in synthetic dataset')
    parser.add_argument('--num_categorical_fields', type=int, default=5, help='Number of categorical fields')
    parser.add_argument('--num_numerical_fields', type=int, default=3, help='Number of numerical fields')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    
    # 모델 관련 인자
    parser.add_argument('--model_name', type=str, default='youtube_ranking', choices=['mmoe', 'youtube_ranking'], 
                        help='Model to use')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--shared_bottom_dims', type=int, nargs='+', default=[512, 256], 
                       help='Shared bottom layer dimensions')
    parser.add_argument('--bottom_mlp_dims', type=int, nargs='+', default=[256, 128], 
                        help='Bottom MLP dimensions')
    parser.add_argument('--tower_mlp_dims', type=int, nargs='+', default=[64, 32], 
                        help='Tower MLP dimensions')
    parser.add_argument('--side_tower_dims', type=int, nargs='+', default=[64, 32], 
                        help='Side tower dimensions')
    parser.add_argument('--task_num', type=int, default=2, help='Number of tasks')
    parser.add_argument('--expert_num', type=int, default=8, help='Number of experts (for MMoE)')
    parser.add_argument('--shared_expert_num', type=int, default=4, help='Number of shared experts (for YouTube Ranking)')
    parser.add_argument('--task_expert_num', type=int, default=2, help='Number of task experts (for YouTube Ranking)')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--use_wide_and_deep', action='store_true', help='Whether to use Wide & Deep architecture')
    
    # 학습 관련 인자
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    
    # 기타 인자
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 시드 설정
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    main(args) 