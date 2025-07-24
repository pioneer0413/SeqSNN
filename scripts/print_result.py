import os
import argparse
import json

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Print the result of a script execution.")
    parser.add_argument("--root_dir_path", type=str, help="Root directory path where the result file is located.",
                        default='/home/hwkang/SeqSNN/warehouse/cluster')
    
    args = parser.parse_args()

    dir_names = os.listdir(args.root_dir_path)
    dir_names.sort()

    _result_path = 'checkpoints/res.json'

    result_file_paths = [os.path.join(args.root_dir_path, dir_name, _result_path) for dir_name in dir_names]
    
    print("Test RSE Results:")
    for i, (dir_name, result_file_path) in enumerate(zip(dir_names, result_file_paths)):
        if os.path.exists(result_file_path):
            with open(result_file_path, 'r') as file:
                result = json.load(file)
                # if not result['test']['rse'] exist, then result['test']['rrse']
                test_rse = result['test'].get('rse', result['test'].get('rrse'))

                print(f"{dir_name}: {test_rse:.4f}")