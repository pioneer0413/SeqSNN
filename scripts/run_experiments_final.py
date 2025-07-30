'''
SeqSNN 실행 지원 스크립트
'''

import subprocess
import sys
import os
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import yaml

config_root_dir = 'exp/forecast'

def load_config(use_cluster, method, dataset_name):
    if use_cluster:
        config_path = f'{config_root_dir}/cluster/{method}_cluster_{dataset_name}.yml'
    else:
        config_path = f'{config_root_dir}/{method}/{method}_{dataset_name}.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config, config_path

def generate_single_command_cluster(config_path, method, dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster, patience, use_all_zero=False, use_all_random=False):
    
    output_dir = f'./warehouse/cluster/{method}_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}_n_cluster={n_cluster}'

    result_path = f'{output_dir}/checkpoints/res.json'

    if os.path.exists(result_path):
        print(f"이미 결과가 존재합니다: {result_path}")
        return None

    cmd = [
        sys.executable, '-m', 'SeqSNN.entry.tsforecast',
        config_path,
        f'--network.encoder_type={encoder_type}',
        f'--network.gpu_id={gpu_id}',
        f'--data.horizon={horizon}',
        f'--runtime.seed={seed}',
        f'--network.n_cluster={n_cluster}',
        f'--runner.early_stop={patience}',
        f'--runtime.output_dir={output_dir}'
    ]

    assert not (use_all_zero and use_all_random), "use_all_zero와 use_all_random은 동시에 사용할 수 없습니다."
    if use_all_zero:
        cmd.append('--network.use_all_zero=True')
    if use_all_random:
        cmd.append('--network.use_all_random=True')

    return cmd

def generate_single_command_baseline(config_path, method, dataset_name, encoder_type, horizon, seed, patience, more_step=False):
    
    output_dir = f'./warehouse/baseline/{method}_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}' if not more_step else f'--runtime.output_dir=./warehouse/baseline/{method}_{dataset_name}_encoder={encoder_type}_horizon={horizon}_baseline_seed={seed}_num_steps=7'

    result_path = f'{output_dir}/checkpoints/res.json'

    if os.path.exists(result_path):
        print(f"이미 결과가 존재합니다: {result_path}")
        return None

    cmd = [
        sys.executable, '-m', 'SeqSNN.entry.tsforecast',
        config_path,
        f'--network.encoder_type={encoder_type}',
        f'--data.horizon={horizon}',
        f'--runtime.seed={seed}',
        f'--runner.early_stop={patience}',
        f'--runtime.output_dir={output_dir}' 
    ]

    if method == 'ispikformer':
        cmd.append(f'--runner.out_size={horizon}')

    if more_step:
        cmd.append('--network.num_steps=7') # TODO: 입력을 받아 처리하도록 수정

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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='SeqSNN 실험 실행 스크립트')
    
    '''
    명령행 인자 정의
    '''
    # 실행 환경
    parser.add_argument('--gpu_ids', type=str, nargs='+', default=['0'])
    parser.add_argument('--max_workers', type=int, default=6)

    # 런타임
    parser.add_argument('--methods', type=str, nargs='+', default=['spikernn', 'spiketcn', 'ispikformer'])
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['electricity', 'solar'])
    parser.add_argument('--encoder_types', type=str, nargs='+', default=['repeat', 'delta', 'conv'])
    parser.add_argument('--horizons', type=int, nargs='+', default=[6, 24, 48, 96])
    parser.add_argument('--seeds', type=int, nargs='+', default=[356, 5857])
    parser.add_argument('--patience_common', type=int, default=20)
    parser.add_argument('--patience_electricity', type=int, default=20) # patience_common과 다르다면 현재 인자 사용
    parser.add_argument('--patience_solar', type=int, default=20) # patience_common과 다르다면 현재 인자 사용
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--batch_size_electricity', type=int, default=64)  # 전력 데이터셋 배치 크기
    parser.add_argument('--batch_size_solar', type=int, default=64)  #
    
    # 클러스터 관련
    parser.add_argument('--use_cluster', action='store_true', default=False)
    parser.add_argument('--use_all_zero', action='store_true', default=False)
    parser.add_argument('--use_all_random', action='store_true', default=False)
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[3])

    # 베이스라인
    parser.add_argument('--more_step', action='store_true', default=False)

    # 스크립트만 생성
    parser.add_argument('--script_only', action='store_true', default=False, help='스크립트만 생성하고 실행하지 않음')

    args = parser.parse_args()

    # <<< 명령행 인자 정의 끝

    # methods, dataset_names, encoder_types, horizons, seeds, n_clusters를 조합하여 모든 실험 조합 생성
    combinations = list(product(
        args.methods,
        args.dataset_names,
        args.encoder_types,
        args.horizons,
        args.seeds,
        args.gpu_ids,
        args.n_clusters
    ))

    # 명령행 인자 출력
    total_experiments = len(combinations)
    print(f"총 실험 조합 수: {total_experiments}")
    print(f"사용할 GPU IDs: {args.gpu_ids}")
    print(f"실험 방법: {args.methods}")
    print(f"데이터셋: {args.dataset_names}")
    print(f"인코더 타입: {args.encoder_types}")
    print(f"예측 지평선: {args.horizons}")
    print(f"시드: {args.seeds}")
    print(f"클러스터 수: {args.n_clusters}")
    print(f"최대 동시 실행 작업 수: {args.max_workers}")
    print(f"공통 조기 중단 인자: {args.patience_common}")
    print(f"전력 데이터셋 조기 중단 인자: {args.patience_electricity}")
    print(f"태양광 데이터셋 조기 중단 인자: {args.patience_solar}")
    print(f"배치 크기: {args.batch_size}")
    print(f"전력 데이터셋 배치 크기: {args.batch_size_electricity}")
    print(f"태양광 데이터셋 배치 크기: {args.batch_size_solar}")
    print(f"추가 단계 사용: {args.more_step}")
    print(f"모든 제로 사용: {args.use_all_zero}")
    print(f"모든 랜덤 사용: {args.use_all_random}")
    print(f"클러스터 사용: {args.use_cluster}")
    input("실험을 시작하려면 Enter 키를 누르세요...")

    commands = []
    for method, dataset_name, encoder_type, horizon, seed, gpu_id, n_cluster in combinations:
        if args.use_cluster:
            if dataset_name == 'electricity':
                patience = args.patience_electricity if args.patience_electricity != args.patience_common else args.patience_common
            elif dataset_name == 'solar':
                patience = args.patience_solar if args.patience_solar != args.patience_common else args.patience_common
            config, config_path = load_config(args.use_cluster, method, dataset_name)
            cmd = generate_single_command_cluster(
                config_path, method, dataset_name, encoder_type, gpu_id, horizon, seed, n_cluster, patience,
                args.use_all_zero, args.use_all_random
            )
        else:
            if dataset_name == 'electricity':
                patience = args.patience_electricity if args.patience_electricity != args.patience_common else args.patience_common
            elif dataset_name == 'solar':
                patience = args.patience_solar if args.patience_solar != args.patience_common else args.patience_common
            config, config_path = load_config(args.use_cluster, method, dataset_name)
            cmd = generate_single_command_baseline(
                config_path, method, dataset_name, encoder_type, horizon, seed, patience, args.more_step
            )

        if cmd is None:
            continue
        
        config_batch_size = config['runner']['batch_size']
        if dataset_name == 'electricity':
            if args.batch_size != args.batch_size_electricity:
                cmd.append(f'--runner.batch_size={args.batch_size_electricity}')
        elif dataset_name == 'solar':
            if args.batch_size != args.batch_size_solar:
                cmd.append(f'--runner.batch_size={args.batch_size_solar}')
        else:
            if args.batch_size != config_batch_size:
                cmd.append(f'--runner.batch_size={args.batch_size}')

        commands.append(cmd)

    
    if args.script_only:
        # 스크립트만 생성
        script_path = 'scripts/run_experiments.sh'
        with open(script_path, 'w') as script_file:
            for cmd in commands:
                script_file.write(' '.join(cmd) + '\n')
        print(f"실험 스크립트가 생성되었습니다: {script_path}")
        #print("스크립트를 실행하려면 다음 명령어를 사용하세요: bash run_experiments.sh")

    else:
        # 병렬로 실험 실행
        run_parallel_experiments(commands, args.max_workers)