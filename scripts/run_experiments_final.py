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
        config_path = f'{config_root_dir}/baseline/{method}_{dataset_name}.yml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config, config_path

def generate_single_command_cluster(config_path, method, dataset_name, encoder_type, horizon, n_cluster, d_model, beta, seed, postfix, patience, gpu_id, use_all_zero=False, use_all_random=False):
    
    output_dir = f'./warehouse/cluster/{method}_{dataset_name}_encoder={encoder_type}_horizon={horizon}_n_cluster={n_cluster}_d_model={d_model}_beta={beta}_seed={seed}_p={postfix}'

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
        f'--network.d_model={d_model}',
        f'--runner.beta={beta}',
    ]

    assert not (use_all_zero and use_all_random), "use_all_zero와 use_all_random은 동시에 사용할 수 없습니다."
    if use_all_zero:
        cmd.append('--network.use_all_zero=True')
        output_dir += '_all-zero'
    if use_all_random:
        cmd.append('--network.use_all_random=True')
        output_dir += '_all-random'

    cmd.append(f'--runtime.output_dir={output_dir}')

    return cmd

def generate_single_command_baseline(config_path, method, dataset_name, encoder_type, horizon, seed, postfix, patience, num_steps):
    
    output_dir = f'./warehouse/baseline/{method}_{dataset_name}_encoder={encoder_type}_horizon={horizon}_seed={seed}_p={postfix}'

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
        f'--network.num_steps={num_steps}',
    ]

    if method == 'ispikformer':
        cmd.append(f'--runner.out_size={horizon}')

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
    parser.add_argument('--architectures', type=str, nargs='+', default=['spikformer']) # ['spikernn', 'spikegru', 'spikformer', 'ispikformer', 'spiketcn', 'snn']
    parser.add_argument('--dataset_names', type=str, nargs='+', default=['electricity', 'solar']) # ['electricity', 'solar', 'metr-la', 'pems-bay']
    parser.add_argument('--encoder_types', type=str, nargs='+', default=['repeat', 'delta', 'conv'])
    parser.add_argument('--horizons', type=int, nargs='+', default=[6, 24, 48, 96])
    parser.add_argument('--seeds', type=int, nargs='+', default=[356, 5857])
    
    parser.add_argument('--patience_electricity', type=int, default=20) # patience_common과 다르다면 현재 인자 사용
    parser.add_argument('--patience_solar', type=int, default=20) # patience_common과 다르다면 현재 인자 사용
    parser.add_argument('--patience_metr-la', type=int, default=20) # patience_common과 다르다면 현재 인자 사용
    parser.add_argument('--patience_pems-bay', type=int, default=20)
    
    parser.add_argument('--batch_size_electricity', type=int, default=64)  # 전력 데이터셋 배치 크기
    parser.add_argument('--batch_size_solar', type=int, default=64)  #
    parser.add_argument('--batch_size_metr-la', type=int, default=64)  # Metr-la 데이터셋 배치 크기
    parser.add_argument('--batch_size_pems-bay', type=int, default=64)

    # 클러스터 관련
    parser.add_argument('--use_cluster', action='store_true', default=False)
    parser.add_argument('--use_all_zero', action='store_true', default=False)
    parser.add_argument('--use_all_random', action='store_true', default=False)
    parser.add_argument('--n_clusters', type=int, nargs='+', default=[3])
    parser.add_argument('--d_model', type=int, nargs='+', default=[512])  # 클러스터링 모델의 차원
    parser.add_argument('--beta', type=float, nargs='+', default=[2e-6])  # 클러스터링 모델의 손실의 비중

    # 베이스라인
    parser.add_argument('--num_steps', type=int, default=4)
    parser.add_argument('--more_steps', type=int, default=3, help='추가 단계')

    # 포스트픽스
    parser.add_argument('--postfix', type=str, default='unknown', help='실험 결과 디렉터리의 포스트픽스')

    # 스크립트만 생성
    parser.add_argument('--script_only', action='store_true', default=False, help='스크립트만 생성하고 실행하지 않음')

    args = parser.parse_args()

    # <<< 명령행 인자 정의 끝

    # architectures, dataset_names, encoder_types, horizons, seeds, n_clusters를 조합하여 모든 실험 조합 생성
    combinations = list(product(
        args.architectures,
        args.dataset_names,
        args.encoder_types,
        args.horizons,
        args.n_clusters,
        args.d_model,
        args.beta,
        args.seeds,
        #args.postfix
    ))

    # 명령행 인자 출력
    total_experiments = len(combinations)
    print(f"총 실험 조합 수: {total_experiments}")
    print(f"사용할 GPU IDs: {args.gpu_ids}")
    print(f"실험 방법: {args.architectures}")
    print(f"데이터셋: {args.dataset_names}")
    print(f"인코더 타입: {args.encoder_types}")
    print(f"예측 지평선: {args.horizons}")
    print(f"클러스터 수: {args.n_clusters}")
    print(f"클러스터링 모델 차원: {args.d_model}")
    print(f"클러스터링 모델 손실 비중: {args.beta}")
    print(f"시드: {args.seeds}")
    print(f"포스트픽스: {args.postfix}")
    print(f"추가 단계 사용: {args.more_steps}")
    print(f"모든 제로 사용: {args.use_all_zero}")
    print(f"모든 랜덤 사용: {args.use_all_random}")
    print(f"클러스터 사용: {args.use_cluster}")
    print(f"최대 동시 실행 작업 수: {args.max_workers}")
    print(f"Electricity 데이터셋 조기 중단 인자: {args.patience_electricity}")
    print(f"Solar 데이터셋 조기 중단 인자: {args.patience_solar}")
    print(f"Metr-la 데이터셋 조기 중단 인자: {args.patience_metr_la}")
    print(f"Pems-bay 데이터셋 조기 중단 인자: {args.patience_pems_bay}")
    print(f"Electricity 데이터셋 배치 크기: {args.batch_size_electricity}")
    print(f"Solar 데이터셋 배치 크기: {args.batch_size_solar}")
    print(f"Metr-la 데이터셋 배치 크기: {args.batch_size_metr_la}")
    print(f"Pems-bay 데이터셋 배치 크기: {args.batch_size_pems_bay}")
    print(f"스크립트만 생성: {args.script_only}")
    
    # y, Y 또는 엔터를 누르면 실험 시작 그 외는 종료
    confirm = input("실험을 시작하시겠습니까? (y/Y 또는 Enter를 누르세요): ")
    if confirm.lower() != 'y' and confirm != '':
        print("실험이 취소되었습니다.")
        sys.exit(0)

    # gpu_ids는 원소를 반복하면서 total_experiments 수 만큼으로 리스트 생성(예: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...])
    gpu_ids = (args.gpu_ids * (total_experiments // len(args.gpu_ids) + 1))[:total_experiments]

    commands = []
    for (method, dataset_name, encoder_type, horizon, n_cluster, d_model, beta, seed), gpu_id in zip(combinations, gpu_ids):
        if args.use_cluster:
            if dataset_name == 'electricity':
                patience = args.patience_electricity
            elif dataset_name == 'solar':
                patience = args.patience_solar
            elif dataset_name == 'metr-la':
                patience = args.patience_metr_la
            elif dataset_name == 'pems-bay':
                patience = args.patience_pems_bay

            config, config_path = load_config(args.use_cluster, method, dataset_name)
            cmd = generate_single_command_cluster(
                config_path, method, dataset_name, encoder_type, horizon, n_cluster, d_model, beta, seed, args.postfix, patience, gpu_id, 
                args.use_all_zero, args.use_all_random
            )
        else:
            if dataset_name == 'electricity':
                patience = args.patience_electricity
            elif dataset_name == 'solar':
                patience = args.patience_solar
            elif dataset_name == 'metr-la':
                patience = args.patience_metr_la
            elif dataset_name == 'pems-bay':
                patience = args.patience_pems_bay

            config, config_path = load_config(args.use_cluster, method, dataset_name)

            # num_steps 설정
            num_steps = (args.num_steps + args.more_steps) if args.more_steps > 0 else args.num_steps
            if num_steps != args.num_steps:
                postfix = f'num_steps={num_steps}-{args.postfix}'

            cmd = generate_single_command_baseline(
                config_path, method, dataset_name, encoder_type, horizon, seed, postfix, patience, num_steps
            )

        if cmd is None:
            continue
        
        if dataset_name == 'electricity':
                cmd.append(f'--runner.batch_size={args.batch_size_electricity}')
        elif dataset_name == 'solar':
                cmd.append(f'--runner.batch_size={args.batch_size_solar}')
        elif dataset_name == 'metr-la':
                cmd.append(f'--runner.batch_size={args.batch_size_metr_la}')
        elif dataset_name == 'pems-bay':
                cmd.append(f'--runner.batch_size={args.batch_size_pems_bay}')
        
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