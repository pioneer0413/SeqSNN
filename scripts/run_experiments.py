#!/usr/bin/env python3
"""
간단한 SeqSNN 실험 실행 스크립트 (병렬 실행)
"""

import subprocess
import sys
import os
from itertools import product
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

target_algorithm = 'spikernn'
patience = 200

"""
모든 실험 조합을 병렬로 실행
"""
# 변수 정의
dataset_names = ['electricity']
encoder_types = ['repeat', 'delta', 'conv']
horizons = [6, 96]
seeds = [707, 808]

max_workers = 6  # 최대 동시 실행 작업 수

def run_single_experiment(dataset_name, encoder_type, horizon, seed, experiment_id, total_experiments):
    """
    단일 실험을 실행하는 함수
    """
    print(f"\n[{experiment_id}/{total_experiments}] 실험 시작: {dataset_name} + {encoder_type}")
    
    # 명령어 구성
    if target_algorithm == 'spikernn':
        cmd = [
            sys.executable, '-m', 'SeqSNN.entry.tsforecast',
            f'exp/forecast/spikernn/spikernn_{dataset_name}.yml',
            f'--network.encoder_type={encoder_type}',
            f'--data.horizon={horizon}',
            f'--runtime.seed={seed}',
            f'--runner.early_stop={patience}',
            f'--runtime.output_dir=./warehouse/patience=200/spikernn_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}'
        ]
    elif target_algorithm == 'spiketcn':
        cmd = [
            sys.executable, '-m', 'SeqSNN.entry.tsforecast',
            f'exp/forecast/tcn/spiketcn2d_{dataset_name}.yml',
            f'--network.encoder_type={encoder_type}',
            f'--data.horizon={horizon}',
            f'--runtime.seed={seed}',
            f'--runtime.output_dir=./warehouse/with_pe/spiketcn_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}'
        ]
    elif target_algorithm == 'ispikformer':
        cmd = [
            sys.executable, '-m', 'SeqSNN.entry.tsforecast',
            f'exp/forecast/ispikformer/ispikformer_{dataset_name}.yml',
            f'--network.encoder_type={encoder_type}',
            f'--data.horizon={horizon}',
            f'--runtime.seed={seed}',
            f'--runtime.output_dir=./warehouse/with_pe/ispikformer_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}'
        ]
    elif target_algorithm == 'snn':
        cmd = [
            sys.executable, '-m', 'SeqSNN.entry.tsforecast',
            f'exp/forecast/snn/snn2d_{dataset_name}.yml',
            f'--network.encoder_type={encoder_type}',
            f'--data.horizon={horizon}',
            f'--runtime.seed={seed}',
            f'--runtime.output_dir=./warehouse/with_pe/snn_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}'
        ]
    
    print(f"명령어: {' '.join(cmd)}")
    
    try:
        # 실험 실행
        result = subprocess.run(
            cmd,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            print(f"✓ 실험 성공: {dataset_name} + {encoder_type}")
            return True
        else:
            print(f"✗ 실험 실패: {dataset_name} + {encoder_type} (코드: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"✗ 실험 오류: {dataset_name} + {encoder_type} - {str(e)}")
        return False


def run_experiments():
    
    # 실험 조합 생성
    experiments = list(product(dataset_names, encoder_types, horizons, seeds))
    
    print(f"총 {len(experiments)}개의 실험을 병렬로 실행합니다.")
    print(f"최대 동시 실행 작업 수: {max_workers}")
    
    success_count = 0
    fail_count = 0
    
    # ThreadPoolExecutor를 사용한 병렬 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 실험 제출
        future_to_experiment = {
            executor.submit(run_single_experiment, dataset_name, encoder_type, horizon, seed, i+1, len(experiments)): 
            (dataset_name, encoder_type, horizon, seed)
            for i, (dataset_name, encoder_type, horizon, seed) in enumerate(experiments)
        }
        
        # 완료된 실험 처리
        for future in as_completed(future_to_experiment):
            dataset_name, encoder_type, horizon, seed = future_to_experiment[future]
            try:
                success = future.result()
                if success:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                print(f"✗ 실험 처리 중 오류: dataset:{dataset_name} + encoder:{encoder_type} + horizon:{horizon} + seed:{seed} - {str(e)}")
                fail_count += 1
    
    # 결과 요약
    print(f"\n{'='*50}")
    print("실험 결과 요약")
    print(f"{'='*50}")
    print(f"총 실험 수: {len(experiments)}")
    print(f"성공: {success_count}")
    print(f"실패: {fail_count}")
    print(f"성공률: {success_count/len(experiments)*100:.1f}%")

if __name__ == "__main__":

    '''
    시작하기 전, 현재 스크립트의 모든 설정값 확인하고 함수 실행
    '''
    print("실험을 시작합니다...")
    # 현재 스크립트의 설정값 출력
    print(f"타겟 알고리즘: {target_algorithm}")
    print(f"조기 중단 기준: {patience} 에폭")
    print("실험 조합:")
    print(f"데이터셋: {dataset_names}")
    print(f"인코더 타입: {encoder_types}")
    print(f"호라이즌: {horizons}")
    print(f"시드: {seeds}")
    print("병렬 실행 최대 작업 수:", max_workers)
    print("========================================")
    # stdin으로 실행 여부 확인
    input("계속하려면 Enter 키를 누르세요...")

    run_experiments()
