'''
Module: run_experiments_spiking.py
Author: Kang Hyun Woo
Last Modified: 2025-09-11 13:00
Description: SeqSNN의 다양한 시계열 예측 실험을 병렬로 실행하는 스크립트 (non-spiking 모델용)
'''

import subprocess
import sys
import os
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import yaml
import signal

source = 'source_hpclab' # 실행 전 반드시 로컬 환경에 맞게 설정
config_root_dir = 'exp/forecast'

def load_config(method, dataset_name):
    if method == 'rnn' or method == 'gru':
        config_path = f'{config_root_dir}/combined/rnn2d_{dataset_name}.yml'
    elif method == 'tcn':
        config_path = f'{config_root_dir}/spiketcn/tcn2d_{dataset_name}.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config, config_path

def generate_single_command(config_path, method, dataset_name, horizon, seed, postfix, patience, gpu_id=0):
    
    output_dir = f'./warehouse/{source}/nonspiking/{method}_{dataset_name}_encoder=none_horizon={horizon}_n_cluster=none_d_model=none_beta=none_seed={seed}_p=nonspiking-{postfix}'
    cmd = [
        sys.executable, '-m', 'SeqSNN.entry.tsforecast',
        config_path,
        #f'--network.gpu_id={gpu_id}',
        f'--data.horizon={horizon}',
        f'--runtime.seed={seed}',
        f'--runner.early_stop={patience}',
    ]

    result_path = f'{output_dir}/checkpoints/res.json'

    if os.path.exists(result_path):
        print(f"이미 결과가 존재합니다: {result_path}")
        return None
    
    cmd.append(f'--runtime.output_dir={output_dir}')

    if method == 'gru':
        cmd.append(f'--network.cell_type=gru')

    return cmd

def run_parallel_experiments(commands, max_workers):
    """
    병렬로 실험을 실행하는 함수
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(subprocess.run, cmd, cwd=os.getcwd()): cmd for cmd in commands}
        for future in as_completed(futures):
            cmd = futures[future]
            try:
                result = future.result()
                if result.returncode == 0:
                    print(f"✓ 실험 성공: {' '.join(cmd)}")
                else:
                    print(f"✗ 실험 실패: {' '.join(cmd)} (코드: {result.returncode})")
            except Exception as e:
                print(f"✗ 실험 중 오류 발생: {' '.join(cmd)} ({e})")

def timeout_handler(signum, frame):
    """타임아웃 핸들러"""
    print("\n⏰ 타임아웃! 실험을 자동으로 시작합니다...")
    raise KeyboardInterrupt

def get_user_confirmation(timeout_seconds=30):
    """사용자 확인을 받되, 타임아웃 시 자동으로 진행"""
    print(f"실험을 시작하시겠습니까? (y/Y 또는 Enter를 누르세요, {timeout_seconds}초 후 자동 시작): ", end='', flush=True)
    
    # 타임아웃 설정
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        confirm = input()
        signal.alarm(0)  # 타이머 취소
        
        if confirm.lower() != 'y' and confirm != '':
            print("실험이 취소되었습니다.")
            sys.exit(0)
        return True
        
    except KeyboardInterrupt:
        signal.alarm(0)  # 타이머 취소
        return True  # 타임아웃 시 자동으로 실험 시작

if __name__=="__main__":
    assert source is not None, "source 변수를 로컬 환경에 맞게 설정하세요."

    parser = argparse.ArgumentParser(description='SeqSNN 실험 실행 스크립트')
    
    '''
    명령행 인자 정의
    '''
    # 실행 환경
    parser.add_argument('--gpu_ids', type=str, nargs='+', default=['0'])
    parser.add_argument('--max_workers', type=int, default=2)

    # 런타임
    parser.add_argument('--architectures', type=str, nargs='+', default=['rnn'])
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['electricity']) 
    parser.add_argument('--horizons', type=int, nargs='+', default=[6])
    parser.add_argument('--seeds', type=int, nargs='+', default=[777])
    
    parser.add_argument('--patience_electricity', type=int, default=10)
    parser.add_argument('--patience_solar', type=int, default=10)
    parser.add_argument('--patience_etth1', type=int, default=30)
    parser.add_argument('--patience_etth2', type=int, default=30)
    parser.add_argument('--patience_metr-la', type=int, default=10)
    parser.add_argument('--patience_weather', type=int, default=25)
    
    parser.add_argument('--batch_size_electricity', type=int, default=32)  # 전력 데이터셋 배치 크기
    parser.add_argument('--batch_size_solar', type=int, default=32)  #
    parser.add_argument('--batch_size_etth1', type=int, default=128)  # etth1 데이터셋 배치 크기
    parser.add_argument('--batch_size_etth2', type=int, default=128)  # etth2 데이터셋 배치 크기
    parser.add_argument('--batch_size_metr-la', type=int, default=32)  # Metr-la 데이터셋 배치 크기
    parser.add_argument('--batch_size_weather', type=int, default=64)  # 날씨 데이터셋 배치 크기

    # 포스트픽스
    parser.add_argument('--postfix', type=str, default='unknown', help='실험 결과 디렉터리의 포스트픽스')

    # 스크립트만 생성
    parser.add_argument('--script_only', action='store_true', default=False, help='스크립트만 생성하고 실행하지 않음')
    parser.add_argument('--timeout', type=int, default=30, help='사용자 확인 타임아웃 시간(초), 0이면 타임아웃 없음')

    args = parser.parse_args()

    # <<< 명령행 인자 정의 끝

    # architectures, dataset_names, encoder_types, horizons, seeds, n_clusters를 조합하여 모든 실험 조합 생성
    combinations = list(product(
        args.architectures,
        args.dataset_names,
        args.horizons,
        args.seeds,
    ))

    # << 세팅 출력
    total_experiments = len(combinations)
    print(f'실험 세팅:')
    print(f"사용할 GPU IDs: {args.gpu_ids}")
    print(f"최대 동시 실행 작업 수: {args.max_workers}")
    print(f"대상 아키텍처: {args.architectures}")
    print(f"데이터셋: {args.dataset_names}")
    print(f"예측 지평선: {args.horizons}")
    print(f"시드: {args.seeds}")
    print(f"포스트픽스: {args.postfix}")
    print('*' * 50)
    print('조기 중단 인자 설정:')
    print(f"Electricity 데이터셋 조기 중단 인자:{args.patience_electricity}")
    print(f"Solar 데이터셋 조기 중단 인자:      {args.patience_solar}")
    print(f"Etth1 데이터셋 조기 중단 인자:      {args.patience_etth1}")
    print(f"Etth2 데이터셋 조기 중단 인자:      {args.patience_etth2}")
    print(f"Metr-la 데이터셋 조기 중단 인자:    {args.patience_metr_la}")
    print(f"Weather 데이터셋 조기 중단 인자:    {args.patience_weather}")
    print('*' * 50)
    print('배치 크기 설정:')
    print(f"Electricity 데이터셋 배치 크기:    {args.batch_size_electricity}")
    print(f"Solar 데이터셋 배치 크기:          {args.batch_size_solar}")
    print(f"Etth1 데이터셋 배치 크기:          {args.batch_size_etth1}")
    print(f"Etth2 데이터셋 배치 크기:          {args.batch_size_etth2}")
    print(f"Metr-la 데이터셋 배치 크기:        {args.batch_size_metr_la}")
    print(f"Weather 데이터셋 배치 크기:        {args.batch_size_weather}")
    print('*' * 50)
    print(f"스크립트만 생성: {args.script_only}")
    print(f"사용자 확인 타임아웃: {args.timeout}초")
    print('*' * 50)
    print(f"총 실험 조합 수: {total_experiments}")
    # << 세팅 출력 끝 
    
    # 사용자 확인 (타임아웃 포함)
    if args.timeout > 0:
        get_user_confirmation(args.timeout)
    else:
        # 타임아웃 없이 일반적인 확인
        confirm = input("실험을 시작하시겠습니까? (y/Y 또는 Enter를 누르세요): ")
        if confirm.lower() != 'y' and confirm != '':
            print("실험이 취소되었습니다.")
            sys.exit(0)

    # gpu_ids는 원소를 반복하면서 total_experiments 수 만큼으로 리스트 생성(예: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...])
    gpu_ids = (args.gpu_ids * (total_experiments // len(args.gpu_ids) + 1))[:total_experiments]
    if args.script_only:
        gpu_ids.sort()

    commands = []
    for (method, dataset_name, horizon, seed), gpu_id in zip(combinations, gpu_ids):
        if dataset_name == 'electricity':
            patience = args.patience_electricity
        elif dataset_name == 'solar':
            patience = args.patience_solar
        elif dataset_name == 'metr-la':
            patience = args.patience_metr_la
        elif dataset_name == 'traffic':
            patience = args.patience_traffic
        elif dataset_name == 'weather':
            patience = args.patience_weather
        elif dataset_name == 'etth1':
            patience = args.patience_etth1
        elif dataset_name == 'etth2':
            patience = args.patience_etth2
        else:
            # 에러 발생
            raise ValueError(f"알 수 없는 데이터셋 이름: {dataset_name}")

        config, config_path = load_config(method, dataset_name)

        postfix = args.postfix

        cmd = generate_single_command(
            config_path, method, dataset_name, horizon, seed, postfix, patience, gpu_id
        )

        if cmd is None: # 이미 결과가 존재하여 건너뜀
            continue
        
        if dataset_name == 'electricity':
            cmd.append(f'--runner.batch_size={args.batch_size_electricity}')
        elif dataset_name == 'solar':
            cmd.append(f'--runner.batch_size={args.batch_size_solar}')
        elif dataset_name == 'etth1':
            cmd.append(f'--runner.batch_size={args.batch_size_etth1}')
        elif dataset_name == 'etth2':
            cmd.append(f'--runner.batch_size={args.batch_size_etth2}')
        elif dataset_name == 'metr-la':
            cmd.append(f'--runner.batch_size={args.batch_size_metr_la}')
        elif dataset_name == 'weather':
            cmd.append(f'--runner.batch_size={args.batch_size_weather}')
        
        commands.append(cmd)

    if args.script_only:
        # 스크립트만 생성
        script_path = 'scripts/run_experiments.sh'
        
        with open(script_path, 'w') as script_file:
            for cmd in commands:
                script_file.write(' '.join(cmd) + '; \\' + '\n')
        print(f"실험 스크립트가 생성되었습니다: {script_path}")
    else:
        # 병렬로 실험 실행
        run_parallel_experiments(commands, args.max_workers)