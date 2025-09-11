import os
import sys
from syops import get_model_complexity_info
import yaml
from SeqSNN.dataset.tsforecast import TSMSDataset
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count


from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass
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
    parser.add_argument('--architecture', type=str, required=True, help='Path to the model architecture configuration file.')
    parser.add_argument('--dataset', type=str, default='electricity', choices=['electricity', 'solar', 'metr-la', 'pems-bay'], help='Path to the dataset configuration file.')
    parser.add_argument('--method', type=str, default='baseline', choices=['nonspiking', 'baseline', 'cluster'], help='Use method for energy consumption calculation.')
    parser.add_argument('--horizon', type=int, default=24, help='Forecasting horizon for the dataset.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dataset.')
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of the model.')
    parser.add_argument('--device', type=int, default=0, help='CUDA device to use for computation.')
    parser.add_argument('--tool', type=str, default='estimated_fr', choices=['syops', 'fvcore', 'firing_rate', 'sops', 'estimated_fr'], help='Tool to use for calculating theoretical energy consumption.')
    args = parser.parse_args()

    root_path = '/home/hwkang/SeqSNN/exp/forecast'

    temp_path = os.path.join(root_path, args.method)

    # <<< SNN 백엔드 설정
    if args.architecture in ['spiketcn', 'spikegru']:
        snn_backend = 'snntorch'
    else:
        snn_backend = 'spikingjelly'
    # <<< SNN 백엔드 설정 완료

    if args.method == 'cluster':
        network_config_path = f'{temp_path}/{args.architecture}_cluster_{args.dataset}.yml'
    elif args.method:
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

    if args.tool == 'syops':
        # 사용 중단
        ost = sys.stdout
        x, _ = next(iter(loader))
        L, C = x.shape[1], x.shape[2]
        ops, params = get_model_complexity_info(
            net, (L, C), loader, as_strings=True, print_per_layer_stat=True, ost=ost
        )
        print('{:<30}  {:<8}'.format('Computational complexity OPs:', ops[0]))
        print('{:<30}  {:<8}'.format('Computational complexity ACs:', ops[1]))
        print('{:<30}  {:<8}'.format('Computational complexity MACs:', ops[2]))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        
    elif args.tool == 'fvcore':
        x, _ = next(iter(loader))
        flops = FlopCountAnalysis(net, x.cuda(device=args.device))
        params = parameter_count(net)
        flops_by_module = flops.by_module()
        params_by_module = params if params else {}
        print(flop_count_table(flops, max_depth=3))
        print('-' * 50)
        total_flops = flops.total()
        print(f'Total FLOPs: {total_flops}')
        print(f'Total Parameters: {sum(params_by_module.values())}')
    elif args.tool == 'firing_rate':
        # 사용 중단
        print('사용 중단 상태')
        '''
        out_monitor = OutputMonitor(net, instance=(LIFNode))

        x, _ = next(iter(loader))
        L, C = x.shape[1], x.shape[2]
        x = x.to(device=args.device)
        net(x)
        
        monitored_layers = out_monitor.monitored_layers
        records = out_monitor.records

        simulation_steps = 4

        cluster_spike_count = net.cluster_spike_count if hasattr(net, 'cluster_spike_count') else None
        cluster_spike_shape = net.cluster_spike_shape if hasattr(net, 'cluster_spike_shape') else None

        for idx, (layer, record) in enumerate(zip(monitored_layers, records)):
            if idx == 0:
                # 텐서인 record의 원소 수를 구함
                total_elements = (record.numel() + cluster_spike_shape.numel()) if cluster_spike_shape is not None else record.numel()
                spike_count = (record.sum().item() + cluster_spike_count) if cluster_spike_count is not None else record.sum().item()
                fr = spike_count / total_elements
            else:
                fr = record.sum().item() / record.numel()
                
            print(f'{layer:<25}: {fr:.12f}')
        '''
    elif args.tool == 'estimated_fr':
        if snn_backend == 'spikingjelly':
            out_monitor = OutputMonitor(net, instance=(LIFNode))
        elif snn_backend == 'snntorch':
            out_monitor = SnnTorchOutputMonitor(net, instance=(snntorch.Leaky))

        x, _ = next(iter(loader))
        L, C = x.shape[1], x.shape[2]
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

        if args.method == 'cluster':
            num_all_neurons += net.cluster_spike_shape.numel() if hasattr(net, 'cluster_spike_shape') else 0
            num_spikes += net.cluster_spike_count
            
            print(f'{net.cluster_spike_shape.numel()}/{num_all_neurons} (Cluster Neurons/All Neurons) = {net.cluster_spike_shape.numel()/num_all_neurons:.3f}%')
            print(f'{net.cluster_spike_count}/{num_spikes} (Cluster Spikes/All Spikes) = {net.cluster_spike_count/num_spikes:.3f}%')

        estimated_fr = num_spikes / num_all_neurons

        print(f'Estimated Firing Rate: {estimated_fr:.12f}')