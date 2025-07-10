import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

def run_command(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SeqSNN experiments with different encoder types and timesteps.")
    parser.add_argument('--num_steps_start', type=int, default=9, help='Starting number of timesteps to use in experiments.')
    parser.add_argument('--num_steps_end', type=int, default=15, help='Ending number of timesteps to use in experiments.')
    args = parser.parse_args()
    num_steps_start = args.num_steps_start
    num_steps_end = args.num_steps_end

    commands = []

    for T in range(num_steps_start, num_steps_end+1):
        commands.append(
            f"python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml "
            f"--network.encoder_type=delta --network.num_steps={T} "
            f"--runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T={T}"
        )
        commands.append(
            f"python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml "
            f"--network.encoder_type=repeat --network.num_steps={T} "
            f"--runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T={T}"
        )
        commands.append(
            f"python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml "
            f"--network.encoder_type=conv --network.num_steps={T} "
            f"--runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T={T}"
        )

    max_workers = 4  # 동시에 실행할 병렬 프로세스 수 (서버 사양에 맞춰 조절)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_command, cmd) for cmd in commands]

        for future in as_completed(futures):
            return_code = future.result()
            if return_code != 0:
                print(f"Command failed with return code: {return_code}")
