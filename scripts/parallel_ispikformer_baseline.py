import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

'''
- iSpikformer에 대해 베이스라인 확보를 위한 병렬 수행 스크립트
- 한 번에 한 데이터 세트에 대해서만 병렬 실행
- 한 번에 한 horizon에 대해서만 병렬 실행
- 따라서 데이터 세트 수 x horizon 수 만큼의 반복은 직접 수행
'''

'''
python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_pems-bay.yml \
--network.encoder_type=repeat --data.horizon=24 --runtime.seed=40 \
--runtime.output_dir=/home/hwkang/SeqSNN/warehouse/with_pe/ispikformer_pems-bay_encoder=repeat_horizon=24_baseline_seed=40
'''

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SeqSNN experiments with different encoder types and timesteps.")
    parser.add_argument('--dataset', type=str, default='electricity', choices=['electricity', 'metr-la', 'pems-bay', 'solar'], help='Dataset to use for the experiments.')
    parser.add_argument('--horizon', type=int, default=24, choices=[6, 24, 48, 96], help='Forecasting horizon.')
    parser.add_argument('--seed', type=int, default=40)
    parser.add_argument('--encoder', type=str, default='repeat', choices=['repeat', 'delta', 'conv'], help='Encoder type to use for the experiments.')
    args = parser.parse_args()

    dst_path = '/home/hwkang/SeqSNN/warehouse/with_pe'

    commands = []

    commands.append(
        f"python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_{args.dataset}.yml "
        f"--network.encoder_type={args.encoder} --data.horizon={args.horizon} --runtime.seed={args.seed} --runner.out_size={args.horizon} "
        f"--runtime.output_dir={dst_path}/ispikformer_{args.dataset}_encoder={args.encoder}_horizon={args.horizon}_baseline_seed={args.seed}"
    )

    max_workers = 1  # 동시에 실행할 병렬 프로세스 수 (서버 사양에 맞춰 조절)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]

        for future in as_completed(futures):
            return_code = future.result()
            if return_code != 0:
                print(f"Command failed with return code: {return_code}")
