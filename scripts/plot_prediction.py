import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import yaml
from SeqSNN.dataset.tsforecast import TSMSDataset

def temporal_mae(y_true, y_pred):
    diff = np.abs(y_true - y_pred) # (n_sample, n_channel * horizon)
    v_mae = np.mean(diff, axis=1)  # (n_sample, )
    return v_mae

def temporal_mse(y_true, y_pred):
    diff = (y_true - y_pred) ** 2  # (n_sample, n_channel * horizon)
    v_mse = np.mean(diff, axis=1)  # (n_sample, )
    return v_mse

def get_y_true(dataset):
    '''
    y_true 획득
    '''
    y_trues = []
    for i in range(len(dataset)):
        y_true = dataset[i][1]
        y_trues.append(y_true)
    # reshape to (n_sample, horizon * n_channel) on y_true
    y_true = np.array(y_trues).reshape(len(dataset), -1)
    return y_true

if __name__ == "__main__":
    # Load the data
    parser = argparse.ArgumentParser(description='Plot predictions from a CSV file.')
    parser.add_argument('--root_path', type=str, default='/home/hwkang/SeqSNN/warehouse/cluster',
                        help='Path to the CSV file containing predictions.')
    parser.add_argument('--dir_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='/home/hwkang/SeqSNN/outputs')
    parser.add_argument('--each', action='store_true',
                        help='If set, save each plot separately.')
    parser.add_argument('--dataset', type=str, default='electricity', choices=['electricity', 'metr-la', 'pems-bay', 'solar'])
    parser.add_argument('--compare', action='store_true', default=False,
                        help='If set, compare predictions with original data.')
    parser.add_argument('--compare_path_1', type=str, default=None)
    parser.add_argument('--compare_path_2', type=str, default=None)
    
    args = parser.parse_args()

    prediction_path = 'checkpoints/test_pre.pkl'

    if args.dataset == 'electricity':
        original_path = '/home/hwkang/SeqSNN/data/electricity/electricity.txt'
        config_path = '/home/hwkang/SeqSNN/exp/forecast/dataset/electricity.yml'
    elif args.dataset == 'metr-la':
        original_path = '/home/hwkang/SeqSNN/data/metr-la.h5'
        config_path = '/home/hwkang/SeqSNN/exp/forecast/dataset/metr-la.yml'
    elif args.dataset == 'pems-bay':
        original_path = '/home/hwkang/SeqSNN/data/pems-bay.h5'
        config_path = '/home/hwkang/SeqSNN/exp/forecast/dataset/pems-bay.yml'
    elif args.dataset == 'solar':
        original_path = '/home/hwkang/SeqSNN/data/solar-energy/solar_AL.txt'
        config_path = '/home/hwkang/SeqSNN/exp/forecast/dataset/solar.yml'

    if original_path.endswith('.txt'):
        original_data = pd.read_csv(original_path, header=None)
    elif original_path.endswith('.h5'):
        original_data = pd.read_hdf(original_path, key='df')

    # yml 파일에서 config 로드
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # remove key name 'type' from config
    config['data'].pop('type', None)

    # dataset_name 명시적 설정
    config['data']['dataset_name'] = 'test'

    if args.dir_path is None:
        '''
        다중 파일 시각화 패스
        '''

        dir_names = os.listdir(args.root_path)
        dir_names = [d for d in dir_names if os.path.isdir(os.path.join(args.root_path, d))]
        file_paths = [os.path.join(args.root_path, d, prediction_path) for d in dir_names if os.path.exists(os.path.join(args.root_path, d, prediction_path))]

        # TODO: 구현
    else:
        
        '''
        단일 파일 시각화 패스
        '''
        file_path = os.path.join(args.dir_path, prediction_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Prediction file not found at {file_path}")
        y_pred = pd.read_pickle(file_path)

        '''
        디버깅(임시)
        '''
        #print(y_pred.shape)
        #print(y_pred.head())

        data_length = y_pred.shape[0]
        hc = y_pred.shape[1]
        horizon = file_path.split('_')[4].split('=')[-1]
        n_channel = hc // int(horizon)

        config['data']['horizon'] = int(horizon)

        dataset = TSMSDataset(**config['data'])

        y_true = get_y_true(dataset)

        '''
        MAE, MSE를 활용해 각 시점에 대한 예측 계산
        '''
        y_mae = temporal_mae(y_true, y_pred)
        y_mse = temporal_mse(y_true, y_pred) # (n_sample, )

        '''
        시각화
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(y_mae, label='MAE', color='blue', markersize=3)
        plt.plot(y_mse, label='MSE', color='red', markersize=3, alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Error')
        plt.legend()
        file_name = args.dir_path.split('/')[-1] if args.dir_path else 'all_predictions'
        plt.title(f'{file_name}')
        plt.savefig(os.path.join(args.save_path, f'{file_name}_mae_mse.png'))
        plt.show()

    if args.compare:
        if args.compare_path_1 is None or args.compare_path_2 is None:
            raise ValueError("Both compare_path_1 and compare_path_2 must be provided for comparison.")
        
        compare_path_1 = os.path.join(args.compare_path_1, prediction_path)
        compare_path_2 = os.path.join(args.compare_path_2, prediction_path)
        
        # Load the comparison data
        y_pred_1 = pd.read_pickle(compare_path_1)
        y_pred_2 = pd.read_pickle(compare_path_2)

        # Ensure both datasets have the same shape
        if y_pred_1.shape != y_pred_2.shape:
            raise ValueError("Comparison datasets must have the same shape.")

        horizon_1 = compare_path_1.split('_')[4].split('=')[-1]
        horizon_2 = compare_path_2.split('_')[4].split('=')[-1]

        assert horizon_1 == horizon_2, "Horizon must be the same for both comparison datasets."

        config['data']['horizon'] = int(horizon_1)
        dataset = TSMSDataset(**config['data'])

        y_true = get_y_true(dataset)
        #y_mae_1 = temporal_mae(y_true, y_pred_1)
       # y_mae_2 = temporal_mae(y_true, y_pred_2)
        y_mse_1 = temporal_mse(y_true, y_pred_1)
        y_mse_2 = temporal_mse(y_true, y_pred_2)

        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.plot(y_mse_1, label='w/o Clustering', color='green', markersize=3)
        plt.plot(y_mse_2, label='w/ Clustering', color='orange', markersize=3, alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        file_name = args.compare_path_2.split('/')[-1]
        plt.savefig(os.path.join(args.save_path, f'{file_name}_compare_mse.png'))
        plt.show()