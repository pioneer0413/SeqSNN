'''
Module: theoretical_energy_consumption.py
Author: Kang Hyun Woo
Last Modified: 2025-09-12 11:31
Description: SeqSNN 모델의 이론적 에너지 소비량을 계산하는 스크립트
'''

import os
import sys
from syops import get_model_complexity_info
import yaml

from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count

from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass
from SeqSNN.dataset.tsforecast import TSMSDataset
from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS
@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()

import argparse
from spikingjelly.activation_based.monitor import OutputMonitor
from spikingjelly.activation_based.neuron import LIFNode

import snntorch
from snntorch.functional.probe import OutputMonitor as SnnTorchOutputMonitor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate theoretical energy consumption for SeqSNN models.")
    parser.add_argument('--architecture', type=str, required=True, 
                        choices=['rnn', 'gru', 'tcn', 'itransformer', 'spikernn', 'spikegru', 'spikformer', 'spiketcn'],help='Path to the model architecture configuration file.')
    parser.add_argument('--dataset', type=str, default='electricity', choices=['electricity', 'solar', 'metr-la', 'pems-bay'], help='Path to the dataset configuration file.')
    parser.add_argument('--method', type=str, default='baseline', choices=['nonspiking', 'baseline', 'cluster'], help='Use method for energy consumption calculation.')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon for the dataset.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dataset.')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device to use for computation.')
    args = parser.parse_args()

    root_path = 'exp/forecast'

    if args.architecture in ['rnn', 'gru', 'tcn', 'itransformer']:
        method = 'nonspiking'
    else:
        method = args.method

    temp_path = os.path.join(root_path, method)

    # <<< SNN 백엔드 설정
    if args.architecture in ['spiketcn', 'spikegru']:
        snn_backend = 'snntorch'
    else:
        snn_backend = 'spikingjelly'
    # <<< SNN 백엔드 설정 완료

    if method == 'cluster':
        network_config_path = f'{temp_path}/{args.architecture}_cluster_{args.dataset}.yml'
    elif method:
        network_config_path = f'{temp_path}/{args.architecture}_{args.dataset}.yml'

    # <<< 데이터셋 로딩
    dataset_config_path = f'{root_path}/dataset/{args.dataset}.yml'

    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset_config['data'].pop('type', None)
    dataset_config['data']['dataset_name'] = 'test'
    dataset_config['data']['horizon'] = args.horizon

    dataset = TSMSDataset(**dataset_config['data'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # <<< 데이터셋 로딩 완료

    # <<< 네트워크 로딩
    network_config = SeqSNNConfig.fromfile(network_config_path)
    network_config.network.d_model = args.d_model
    network_config.data.horizon = args.horizon
    network_config.network.gpu_id = args.device
    
    net = network_config.network.build(
        input_size=dataset.num_variables, max_length=dataset.max_seq_len
    )
    net.cuda(device=args.device)
    # <<< 네트워크 로딩 완료

    # <<< commmon
    x, _ = next(iter(loader))
    L, C = x.shape[1], x.shape[2]

    # <<< syops
    '''
    # args.batch_size가 32 보다 작은 경우 시간이 너무 오래 걸릴 수 있다는 경고 메시지 출력
    if args.batch_size < 32:
        print("⚠️ 경고: batch_size가 32보다 작으면 syops 계산에 시간이 오래 걸릴 수 있습니다.")
        # 사용자에게 계속 진행할지 묻기
        proceed = input("계속 진행하시겠습니까? ([y]/n): ")
        if proceed.lower() != 'y':
            print("프로그램을 종료합니다.")
            sys.exit(0)

    ost = sys.stdout
    ops, params = get_model_complexity_info(
        net, (L, C), loader, as_strings=True, print_per_layer_stat=True, ost=ost
    )
    '''

    '''
    print('{:<30}  {:<8}'.format('Computational complexity OPs:', ops[0]))
    print('{:<30}  {:<8}'.format('Computational complexity ACs:', ops[1]))
    print('{:<30}  {:<8}'.format('Computational complexity MACs:', ops[2]))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''

    #syops_ops = ops[0]
    #syops_acs = ops[1]
    #syops_macs = ops[2]
    #syops_params = params

    # <<< fvcore
    flops = FlopCountAnalysis(net, x.cuda(device=args.device))
    params = parameter_count(net)
    flops_by_module = flops.by_module()
    params_by_module = params if params else {}
    #print(flop_count_table(flops, max_depth=3))
    #print('-' * 50)
    total_flops = flops.total()
    total_params = sum(params_by_module.values())
    #print(f'Total FLOPs: {total_flops}')
    #print(f'Total Parameters: {sum(params_by_module.values())}')

    # <<< firing rate
    if method != 'nonspiking':
        if snn_backend == 'spikingjelly':
            out_monitor = OutputMonitor(net, instance=(LIFNode))
        elif snn_backend == 'snntorch':
            out_monitor = SnnTorchOutputMonitor(net, instance=(snntorch.Leaky))

        x = x.to(device=args.device)

        net(x)
        
        monitored_layers = out_monitor.monitored_layers
        records = out_monitor.records

        num_all_neurons = 0
        num_spikes = 0

        for record in records:
            if args.architecture == 'spikegru':
                record = record[0]

            num_neurons = record.numel()
            num_all_neurons += num_neurons
            num_spikes += record.sum().item()

        if method == 'cluster':
            num_all_neurons += net.cluster_spike_shape.numel() if hasattr(net, 'cluster_spike_shape') else 0
            num_spikes += net.cluster_spike_count
        
        # num_spikes를 int형으로 변환
        num_spikes = int(num_spikes)

        estimated_fr = num_spikes / num_all_neurons

    # <<< 결과 전체 출력
    print('=' * 50)
    print('### Settings ###')
    print(f'모델명: {args.architecture}\n데이터셋: {args.dataset}\n방법: {method}\nd_model: {args.d_model}\nhorizon: {args.horizon}\nbatch_size: {args.batch_size}')
    print('=' * 50)
    #print('### Syops ###')
    #print(f'FLOPs (OPs): {syops_ops}')
    #print(f'FLOPs (ACs): {syops_acs}')
    #print(f'FLOPs (MACs): {syops_macs}')
    #print(f'Parameters: {syops_params}')
    #print('=' * 50)
    print('### fvcore ###')
    print(f'Total FLOPs: {total_flops} ({total_flops / L:.2f} FLOPs/step)')
    print(f'Total Parameters: {total_params}')
    print('=' * 50)
    if method != 'nonspiking':
        print('### Firing Rate ###')
        print('Estimated Firing Rate: {:.3f}% ({}/{})'.format(estimated_fr * 100, num_spikes, num_all_neurons))
        print('=' * 50)
