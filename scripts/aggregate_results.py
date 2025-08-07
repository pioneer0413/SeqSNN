import os
import json

'''
warhouse에 있는 모든 디렉터리를 순회하면서 res.json 파일을 찾아 r2와 rse 값을 추출해 CSV 파일에 종합하여 저장
'''

def get_file_existences(dir_path, target_file='res.json'):
    # 1. dir_path 하위의 모든 디렉터리를 탐색해 'res.json' 파일이 있다면, res.json 파일의 경로를 리스트로 반환
    
    res_files = []
    for root, dirs, files in os.walk(dir_path):
        if target_file in files:
            res_files.append(os.path.join(root, target_file))
    return res_files

# 디렉터리 이름 내에 주어진 키워드가 전부 포함되어 있는 지 확인하는 함수
def is_dir_name_contains_keywords(dir_name, keywords):
    return all(keyword in dir_name for keyword in keywords)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate results from res.json files.')
    parser.add_argument('--dir_path', type=str, default="warehouse/baseline", help='Directory path to search for res.json files')
    args = parser.parse_args()

    # Get the list of res.json file paths
    res_files = get_file_existences(args.dir_path)

    # json 파일의 res_files를 읽어서 ['test']['r2']와 ['test']['rse'] 값을 추출
    results = []
    for res_file in res_files:
        with open(res_file, 'r') as f:
            data = json.load(f)
            test_r2 = data['test']['r2']

            # data['test']['rse']가 존재하는 경우에만 추가, 'rse'가 없는 경우 'rrse'를 키로 사용
            if 'rse' in data['test']:
                test_rse = data['test']['rse']
            elif 'rrse' in data['test']:
                test_rse = data['test']['rrse']
            else:
                test_rse = None

            assert test_r2 is not None, f"test['r2'] is None in {res_file}"

            tokens = res_file.split('/')
            source = tokens[1]
            filename = os.path.basename(os.path.dirname(os.path.dirname(res_file)))
            tokens = filename.split('_')

            if len(tokens) < 4:
                print(f"Skipping {res_file} due to insufficient tokens.")
                continue

            # architecture
            architecture = tokens[0]
            
            # dataset
            dataset = tokens[1]
            
            # encoder
            if tokens[2].startswith('encoder='):
                encoder = tokens[2].split('=')[1]
            else:
                encoder = 'unknown'
            
            # horizon
            if tokens[3].startswith('horizon='):
                horizon = tokens[3].split('=')[1]
            else:
                horizon = 'unknown'
            
            # n_cluster
            # filename에 'n_cluster'가 포함되어 있으면, 그 다음 '=' 다음의 값을 n_cluster로 사용
            if 'n_cluster' in filename:
                n_cluster_index = filename.index('n_cluster') + len('n_cluster=')
                n_cluster_end_index = filename.find('_', n_cluster_index)
                if n_cluster_end_index == -1:
                    n_cluster_end_index = len(filename)
                n_cluster = filename[n_cluster_index:n_cluster_end_index]
            else:
                n_cluster = 'unknown'

            # d_model
            # filename에 'd_model'이 포함되어 있으면, 그 다음 '=' 다음의 값을 d_model로 사용
            if 'd_model' in filename:
                d_model_index = filename.index('d_model') + len('d_model=')
                d_model_end_index = filename.find('_', d_model_index)
                if d_model_end_index == -1:
                    d_model_end_index = len(filename)
                d_model = filename[d_model_index:d_model_end_index]
            else:
                d_model = 'unknown'
            
            # beta
            # filename에 'beta'가 포함되어 있으면, 그 다음 '=' 다음의 값을 beta로 사용
            if 'beta' in filename:
                beta_index = filename.index('beta') + len('beta=')
                beta_end_index = filename.find('_', beta_index)
                if beta_end_index == -1:
                    beta_end_index = len(filename)
                beta = filename[beta_index:beta_end_index]
            else:
                beta = 'unknown'

            # 'seed'라는 문자열이 없는 경우 continue
            if len(tokens) < 6 or not tokens[5].startswith('seed='):
                print(f"Skipping {res_file} due to missing 'seed' information.")
                continue

            if tokens[5].startswith('seed='):
                seed = tokens[5].split('=')[1]

            postfix = 'unknown'
            # 'num_steps', 'all_zero', 'all_random'이 포함된 경우 postfix에 추가
            if 'num_steps' in filename: # 'num_steps=<값>'을 추가
                postfix = 'num_steps=' + filename.split('num_steps=')[1].split('_')[0]
            elif 'all_zero' in filename:
                postfix = 'all_zero'
            elif 'all_random' in filename:
                postfix = 'all_random'

            results.append({
                'source': source,
                'architecture': architecture,
                'dataset': dataset,
                'encoder': encoder,
                'horizon': horizon,
                'n_cluster': n_cluster,
                'd_model': d_model,
                'beta': beta,
                'seed': seed,
                'postfix': postfix,
                'r2': test_r2,
                'rse': test_rse,
                'filepath': res_file
            })

            
    # results를 CSV 파일로 저장
    import pandas as pd
    df = pd.DataFrame(results)
    output_file = 'analysis/aggregated_results.csv' #os.path.join(args.dir_path, 'aggregated_results.csv')
    df.to_csv(output_file, index=False)
    print(f"Aggregated results saved to {output_file}")