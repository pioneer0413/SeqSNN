import os
import argparse
import pandas as pd
import numpy as np

src_dir = '/home/hwkang/SeqSNN/dataset_link/forecasting'
dst_dir = '/home/hwkang/SeqSNN/data'

def convert_long_to_wide(df):
    """
    Long format을 Wide format으로 변환하는 함수
    
    Args:
        df: pandas DataFrame with columns ['date', 'data', 'cols']
    
    Returns:
        Wide format DataFrame with date as index and cols as columns
    """
    print("=== 원본 데이터 정보 ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique cols: {df['cols'].unique()}")
    print(f"Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    # date 컬럼을 datetime으로 변환
    df['date'] = pd.to_datetime(df['date'])
    
    # pivot_table을 사용하여 long format을 wide format으로 변환
    # index: date (시간), columns: cols (채널), values: data (값)
    wide_df = df.pivot_table(
        index='date',     # 행 인덱스: 날짜/시간
        columns='cols',   # 열: 채널명 (channel_1, channel_2, ...)
        values='data',    # 값: 데이터 값
        aggfunc='first'   # 중복 값이 있을 경우 첫 번째 값 사용
    )
    
    # 컬럼명 정리 (멀티인덱스 제거)
    wide_df.columns.name = None
    
    print("\n=== 변환된 데이터 정보 ===")
    print(f"Shape: {wide_df.shape}")
    print(f"Columns: {wide_df.columns.tolist()}")
    print(f"Index name: {wide_df.index.name}")
    print(f"Date range: {wide_df.index.min()} ~ {wide_df.index.max()}")
    
    return wide_df

def save_converted_data(wide_df, original_filename, dst_dir, save_format='h5'):
    """변환된 데이터를 저장하는 함수"""
    # 출력 파일명 생성 (original_filename에서 확장자 제거)
    base_name = os.path.splitext(original_filename)[0]
    
    # 디렉터리 생성 (없으면)
    os.makedirs(dst_dir, exist_ok=True)
    
    if save_format == 'txt':
        # TXT 형식: 값만 저장 (컬럼, 인덱스 제외)
        output_filename = f"{base_name}_wide.txt"
        output_path = os.path.join(dst_dir, output_filename)
        
        # 값만 numpy array로 추출하여 CSV 형식으로 저장
        np.savetxt(output_path, wide_df.values, delimiter=',', fmt='%.6f')
        
        print(f"\n✅ TXT 형식으로 저장되었습니다: {output_path}")
        print(f"   - 형식: 값만 저장 (인덱스/컬럼 제외)")
        print(f"   - 구분자: 쉼표(,)")
        print(f"   - 형태: {wide_df.shape[0]}행 × {wide_df.shape[1]}열")
        
    elif save_format == 'h5':
        # H5 형식: tsforecast.py 호환 구조로 저장
        output_filename = f"{base_name}_wide.h5"
        output_path = os.path.join(dst_dir, output_filename)
        
        # tsforecast.py에서 읽는 방식에 맞게 저장
        # pd.read_hdf(file).reset_index() 형태로 읽히도록 구조화
        wide_df.to_hdf(output_path, key='df', mode='w', format='table')
        
        print(f"\n✅ H5 형식으로 저장되었습니다: {output_path}")
        print(f"   - 키: 'df'")
        print(f"   - tsforecast.py 호환 구조")
        print(f"   - 인덱스: 날짜/시간 포함")
        print(f"   - 형태: {wide_df.shape[0]}행 × {wide_df.shape[1]}열")
        
    elif save_format == 'csv':
        # CSV 형식: 기존 방식 유지
        output_filename = f"{base_name}_wide.csv"
        output_path = os.path.join(dst_dir, output_filename)
        
        # CSV로 저장 (인덱스 포함)
        wide_df.to_csv(output_path)
        
        print(f"\n✅ CSV 형식으로 저장되었습니다: {output_path}")
        print(f"   - 인덱스/컬럼 포함")
        print(f"   - 형태: {wide_df.shape[0]}행 × {wide_df.shape[1]}열")
        
    else:
        raise ValueError(f"지원하지 않는 저장 형식: {save_format}. 지원 형식: txt, h5, csv")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert long format CSV to wide format")
    parser.add_argument('--target_file', type=str, required=True, help='Target CSV file name in long format')
    parser.add_argument('--preview', action='store_true', help='Preview the conversion without saving')
    parser.add_argument('--format', type=str, choices=['txt', 'h5', 'csv'], default='txt', 
                        help='Output format: txt (values only), h5 (tsforecast compatible), csv (with index/columns)')
    args = parser.parse_args()

    target_file = args.target_file
    target_file_path = os.path.join(src_dir, target_file)
    
    print(f"🔍 Processing file: {target_file_path}")
    print(f"📁 Output format: {args.format.upper()}")
    
    # CSV 읽기
    try:
        df = pd.read_csv(target_file_path)
        print("✅ 파일을 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {target_file_path}")
        exit(1)
    except Exception as e:
        print(f"❌ 파일 읽기 중 오류 발생: {e}")
        exit(1)
    
    # 필수 컬럼 확인
    required_cols = ['date', 'data', 'cols']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ 필수 컬럼이 없습니다. 필요한 컬럼: {required_cols}")
        print(f"현재 컬럼: {df.columns.tolist()}")
        exit(1)
    
    # Long format을 Wide format으로 변환
    wide_df = convert_long_to_wide(df)
    
    # 미리보기 출력
    print("\n=== 변환 결과 미리보기 ===")
    print("첫 5행:")
    print(wide_df.head())
    print("\n마지막 5행:")
    print(wide_df.tail())
    print(f"\n결측값 개수:")
    print(wide_df.isnull().sum())
    
    # 저장 또는 미리보기만
    if args.preview:
        print("\n🔍 미리보기 모드: 파일을 저장하지 않습니다.")
        print(f"선택된 형식: {args.format.upper()}")
        if args.format == 'txt':
            print("TXT 형식 - 값만 저장됩니다 (인덱스/컬럼 제외)")
        elif args.format == 'h5':
            print("H5 형식 - tsforecast.py 호환 구조로 저장됩니다")
        elif args.format == 'csv':
            print("CSV 형식 - 인덱스/컬럼 포함하여 저장됩니다")
    else:
        # 변환된 데이터 저장
        output_path = save_converted_data(wide_df, target_file, dst_dir, args.format)
        
        print(f"\n📊 변환 완료!")
        print(f"  - 원본: {df.shape[0]} 행 × {df.shape[1]} 열")
        print(f"  - 변환: {wide_df.shape[0]} 행 × {wide_df.shape[1]} 열")
        print(f"  - 저장 위치: {output_path}")
        print(f"  - 저장 형식: {args.format.upper()}")