import os
import pandas as pd

if __name__=='__main__':
    target_file = '/home/hwkang/SeqSNN/analysis/aggregated_results.csv'

    df = pd.read_csv(target_file)

    # 'rse' 열의 값을 사용하여 r2 값을 계산하고 'refined_r2' 열로 추가
    df['refined_r2'] = 1 - df['rse']**2

    # 변경된 DataFrame을 새로운 CSV 파일로 저장하여 확인
    output_file = '/home/hwkang/SeqSNN/analysis/aggregated_results_with_r2.csv'
    df.to_csv(output_file, index=False)

    print(f"'{output_file}'에 'refined_r2' 열이 추가된 파일이 저장되었습니다.")
    print(df[['rse', 'refined_r2']].head())