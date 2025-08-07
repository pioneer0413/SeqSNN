'''
텍스트로 전달된 스크립트를 병렬로 실행
'''
import os
import sys
import subprocess
import threading
import time
from datetime import datetime


def run_command(command, index):
    """개별 명령어를 실행하는 함수"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting command {index+1}")
    
    # 명령어 끝의 세미콜론 제거
    command = command.rstrip(';')
    
    try:
        # subprocess를 사용하여 명령어 실행
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 실시간으로 출력 표시
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"[CMD{index+1}] {output.strip()}")
        
        # 프로세스 종료 대기
        return_code = process.wait()
        
        if return_code == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Command {index+1} completed successfully")
        else:
            stderr = process.stderr.read()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Command {index+1} failed with return code {return_code}")
            if stderr:
                print(f"[CMD{index+1} ERROR] {stderr}")
                
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Command {index+1} failed with exception: {e}")


if __name__ == "__main__":
    
    '''
    여기에 명령어 목록을 추가
    '''
    commands = [
        '/root/miniconda3/envs/SeqSNN/bin/python -m SeqSNN.entry.tsforecast exp/forecast/cluster/spikernn_cluster_metr-la.yml --network.encoder_type=conv --network.gpu_id=0 --data.horizon=6 --runtime.seed=138 --network.n_cluster=3 --runner.early_stop=6 --network.d_model=256 --runner.beta=2e-06 --runtime.output_dir=./warehouse/source_kdg/cluster/spikernn_metr-la_encoder=conv_horizon=6_n_cluster=3_d_model=256_beta=2e-06_seed=138_p=unknown --runner.batch_size=64;',
        '/root/miniconda3/envs/SeqSNN/bin/python -m SeqSNN.entry.tsforecast exp/forecast/cluster/spikernn_cluster_metr-la.yml --network.encoder_type=conv --network.gpu_id=0 --data.horizon=24 --runtime.seed=138 --network.n_cluster=3 --runner.early_stop=6 --network.d_model=256 --runner.beta=2e-06 --runtime.output_dir=./warehouse/source_kdg/cluster/spikernn_metr-la_encoder=conv_horizon=24_n_cluster=3_d_model=256_beta=2e-06_seed=138_p=unknown --runner.batch_size=64;',
        '/root/miniconda3/envs/SeqSNN/bin/python -m SeqSNN.entry.tsforecast exp/forecast/cluster/spikernn_cluster_metr-la.yml --network.encoder_type=conv --network.gpu_id=0 --data.horizon=48 --runtime.seed=138 --network.n_cluster=3 --runner.early_stop=6 --network.d_model=256 --runner.beta=2e-06 --runtime.output_dir=./warehouse/source_kdg/cluster/spikernn_metr-la_encoder=conv_horizon=48_n_cluster=3_d_model=256_beta=2e-06_seed=138_p=unknown --runner.batch_size=64;',
        '/root/miniconda3/envs/SeqSNN/bin/python -m SeqSNN.entry.tsforecast exp/forecast/cluster/spikernn_cluster_metr-la.yml --network.encoder_type=conv --network.gpu_id=0 --data.horizon=96 --runtime.seed=138 --network.n_cluster=3 --runner.early_stop=6 --network.d_model=256 --runner.beta=2e-06 --runtime.output_dir=./warehouse/source_kdg/cluster/spikernn_metr-la_encoder=conv_horizon=96_n_cluster=3_d_model=256_beta=2e-06_seed=138_p=unknown --runner.batch_size=64;',
        '/root/miniconda3/envs/SeqSNN/bin/python -m SeqSNN.entry.tsforecast exp/forecast/cluster/spikernn_cluster_metr-la.yml --network.encoder_type=conv --network.gpu_id=0 --data.horizon=96 --runtime.seed=139 --network.n_cluster=3 --runner.early_stop=6 --network.d_model=256 --runner.beta=2e-06 --runtime.output_dir=./warehouse/source_kdg/cluster/spikernn_metr-la_encoder=conv_horizon=96_n_cluster=3_d_model=256_beta=2e-06_seed=139_p=unknown --runner.batch_size=64;'
    ]
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting parallel execution of {len(commands)} commands")
    
    # 스레드 리스트
    threads = []
    
    # 각 명령어를 별도 스레드에서 실행
    for i, command in enumerate(commands):
        thread = threading.Thread(target=run_command, args=(command, i))
        threads.append(thread)
        thread.start()
        
        # 스레드 시작 간격 (선택사항 - GPU 메모리 경합 방지)
        time.sleep(1)
    
    # 모든 스레드 완료 대기
    for thread in threads:
        thread.join()
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All commands completed")
