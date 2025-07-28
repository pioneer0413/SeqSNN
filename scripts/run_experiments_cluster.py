#!/usr/bin/env python3
"""
SeqSNN 클러스터 실험 실행 스크립트 (병렬 실행)
"""

import subprocess
import sys
import os
from itertools import product
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import argparse

def run_single_experiment(dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster, patience, post_id, experiment_id, total_experiments, use_all_zero=False, use_all_random=False):
    """
    단일 실험을 실행하는 함수
    """
    print(f"\n[{experiment_id}/{total_experiments}] 실험 시작: {dataset_name} + {encoder_type} + GPU{gpu_id} + horizon={horizon} + seed={seed}")

    # 출력 디렉토리 이름 구성
    output_dir_suffix = f"_post_id={post_id}"
    if use_all_zero:
        output_dir_suffix += "_all_zero"
    if use_all_random:
        output_dir_suffix += "_all_random"
    
    # 명령어 구성
    cmd = [
        sys.executable, '-m', 'SeqSNN.entry.tsforecast',
        f'exp/forecast/spikernn/spikernn_cluster_{dataset_name}.yml',
        f'--network.encoder_type={encoder_type}',
        f'--network.gpu_id={gpu_id}',
        f'--data.horizon={horizon}',
        f'--runtime.seed={seed}',
        f'--network.gpu_id={gpu_id}',
        f'--network.n_cluster={n_cluster}',
        f'--runner.early_stop={patience}',
        f'--runtime.output_dir=./warehouse/cluster/spikernn_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}_n_cluster={n_cluster}{output_dir_suffix}'
    ]
    
    # 클러스터 설정 옵션 추가
    if use_all_zero:
        cmd.append('--network.use_all_zero=True')
    if use_all_random:
        cmd.append('--network.use_all_random=True')
    
    print(f"명령어: {' '.join(cmd)}")
    
    try:
        # 실험 실행
        result = subprocess.run(
            cmd,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✓ 실험 성공: {dataset_name} + {encoder_type} + GPU{gpu_id} + horizon={horizon} + seed={seed} + n_cluster={n_cluster}")
            return True
        else:
            print(f"✗ 실험 실패: {dataset_name} + {encoder_type} + GPU{gpu_id} + horizon={horizon} + seed={seed} (코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ 실험 오류: {dataset_name} + {encoder_type} + GPU{gpu_id} + horizon={horizon} + seed={seed} - {str(e)}")
        return False


def run_experiments(dataset_names, gpu_ids, max_workers=None, patience=30, post_id=0, use_all_zero=False, use_all_random=False):
    """
    모든 실험 조합을 병렬로 실행
    """
    # 변수 정의
    #dataset_names = ['electricity', 'metr-la', 'pems-bay', 'solar']
    encoder_types = ['repeat', 'delta', 'conv']
    #cluster_loss_weights = [1e-5, 1e-8, 1e-10]  # 클러스터 손실 가중치
    horizons = [6, 24, 48, 96]  # 예측 호라이즌
    seeds = [222,333]
    cluster_list = [3]
    
    # max_workers를 GPU 개수로 자동 설정 (지정되지 않은 경우)
    if max_workers is None:
        max_workers = len(gpu_ids)
    
    print(f"사용할 GPU IDs: {gpu_ids}")
    #print(f"클러스터 손실 가중치: {cluster_loss_weights}")
    print(f"예측 호라이즌: {horizons}")
    print(f"시드: {seeds}")
    print(f"클러스터: {cluster_list}")
    print(f"최대 동시 실행 작업 수: {max_workers}")
    
    # 실험 조합 생성 (GPU를 순환하며 할당)
    experiments = []
    experiment_id = 1
    
    for seed in seeds:
        for dataset_name in dataset_names:
            for horizon in horizons:
                for encoder_type in encoder_types:                
                    for n_cluster in cluster_list:
                        gpu_id = gpu_ids[(experiment_id - 1) % len(gpu_ids)]
                        experiments.append((dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster))
                        experiment_id += 1

    print(f"총 {len(experiments)}개의 실험을 병렬로 실행합니다.")

    #print(experiments)
    
    success_count = 0
    fail_count = 0
    
    # ThreadPoolExecutor를 사용한 병렬 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 실험 제출
        future_to_experiment = {
            executor.submit(run_single_experiment, dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster, patience, post_id, i+1, len(experiments), use_all_zero, use_all_random): 
            (dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster)
            for i, (dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster) in enumerate(experiments)
        }
        
        # 완료된 실험 처리
        for future in as_completed(future_to_experiment):
            dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster = future_to_experiment[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"✗ 실험 처리 중 오류: {dataset_name} + {encoder_type} + GPU{gpu_id} + horizon={horizon} + seed={seed} + n_cluster={n_cluster} - {str(e)}")
                fail_count += 1
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("실험 결과 요약")
    print(f"{'='*50}")
    print(f"총 실험 수: {len(experiments)}")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")
    print(f"성공률: {success_count/len(experiments)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='SeqSNN 클러스터 실험 실행')
    parser.add_argument('--gpu_ids', type=str, required=True, 
                        help='사용할 GPU ID들을 쉼표로 구분하여 입력 (예: 0,1,2)')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='최대 동시 실행 작업 수 (기본값: GPU 개수)')
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--post_id', type=int, default=0,)
    parser.add_argument('--dataset_names', type=str, default='electricity,metr-la,pems-bay,solar',
                        help='실험에 사용할 데이터셋 이름들을 쉼표로 구분하여 입력 (예: electricity,metr-la,pems-bay,solar)')
    parser.add_argument('--use_all_zero', action='store_true', default=False,
                        help='클러스터 확률을 모두 0으로 설정')
    parser.add_argument('--use_all_random', action='store_true', default=False,
                        help='클러스터 확률을 모두 랜덤으로 설정')
    
    args = parser.parse_args()
    
    # GPU ID 파싱
    try:
        gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(',')]
        dataset_names = [name.strip() for name in args.dataset_names.split(',')]
    except ValueError:
        print("오류: GPU ID는 숫자여야 합니다 (예: 0,1,2)")
        sys.exit(1)
    
    if not gpu_ids:
        print("오류: 최소 하나의 GPU ID를 지정해야 합니다")
        sys.exit(1)
    
    print(f"GPU IDs: {gpu_ids}")
    print(f"Max workers: {args.max_workers if args.max_workers else len(gpu_ids)}")
    print(f"Use all zero: {args.use_all_zero}")
    print(f"Use all random: {args.use_all_random}")
    
    # 실험 실행
    run_experiments(dataset_names, gpu_ids, args.max_workers, args.patience, args.post_id, args.use_all_zero, args.use_all_random)


if __name__ == "__main__":
    main()
