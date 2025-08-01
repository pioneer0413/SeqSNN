import os
import argparse
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Print the result of a script execution.")
    parser.add_argument("--root_dir_path", type=str, help="Root directory path where the result file is located.",
                        default='/home/hwkang/SeqSNN/warehouse/cluster')
    parser.add_argument("--metric", type=str, choices=['rse', 'r2', 'both'], default='rse')
    
    args = parser.parse_args()

    dir_names = os.listdir(args.root_dir_path)
    dir_names.sort()

    _result_path = 'checkpoints/res.json'

    result_file_paths = [os.path.join(args.root_dir_path, dir_name, _result_path) for dir_name in dir_names]

    algorithm_names = ['spikernn', 'spiketcn', 'ispikformer', 'spikformer']
    dataset_names = ['electricity', 'solar']
    encoder_names = ['conv', 'delta', 'repeat']
    
    patterns = [f"{algo}_{dataset}_encoder={encoder}" for algo in algorithm_names for dataset in dataset_names for encoder in encoder_names]

    # result_file_paths에서 patterns로 그룹핑
    pattern_results = {pattern: [] for pattern in patterns}
    for i, (dir_name, result_file_path) in enumerate(zip(dir_names, result_file_paths)):
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as file:
                result = json.load(file)
                # metric에 따라 결과값 선택
                if args.metric == 'rse':
                    test_metric = result['test'].get('rse', result['test'].get('rrse'))
                elif args.metric == 'r2':
                    test_metric = result['test'].get('r2')
                elif args.metric == 'both':
                    test_rse = result['test'].get('rse', result['test'].get('rrse'))
                    test_r2 = result['test'].get('r2')
                    test_metric = (test_rse, test_r2)
                
                for pattern in patterns:
                    if pattern in dir_name:
                        horizon = dir_name.split('_')[3].split('=')[1]
                        horizon = f'{int(horizon):02}'
                        #seed = dir_name.split('_')[-1].split('=')[1]
                        pattern_results[pattern].append((dir_name, test_metric, horizon))
                        # sort by horizon
                        pattern_results[pattern].sort(key=lambda x: (x[2]))
    
    print(f"Test {args.metric.upper()} Results:")
    for pattern, results in pattern_results.items():
        print(f"Results for {pattern}:")
        for dir_name, test_metric, horizon in results:
            if args.metric == 'both':
                rse_val, r2_val = test_metric
                print(f"  {dir_name}: RSE= {rse_val:.4f}, R²= {r2_val:.4f} (horizon: {horizon})")
            else:
                print(f"  {dir_name}: {test_metric:.4f} (horizon: {horizon})")