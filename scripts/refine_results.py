import os
import json
import shutil

'''
warhouse에 있는 모든 디렉터리를 순회하면서 res.json 파일을 찾아 r2와 rse 값을 추출해 CSV 파일에 종합하여 저장
디렉터리명에서 query_string을 target_string으로 교체하여 디렉터리명을 변경
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

def rename_directories_with_query_string(dir_path, query_string, target_string, dry_run=True):
    """
    디렉터리명에서 query_string을 target_string으로 교체하여 디렉터리명을 변경
    
    Args:
        dir_path: 검색할 디렉터리 경로
        query_string: 찾을 문자열
        target_string: 교체할 문자열
        dry_run: True이면 실제로 변경하지 않고 변경될 내용만 출력
    
    Returns:
        변경된 디렉터리 수
    """
    renamed_count = 0
    
    print(f"🔍 '{dir_path}' 디렉터리에서 '{query_string}'을 '{target_string}'으로 교체할 디렉터리 검색 중...")
    
    # 모든 디렉터리를 탐색
    for root, dirs, files in os.walk(dir_path, topdown=False):  # topdown=False로 하위 디렉터리부터 처리
        for dir_name in dirs:
            if query_string in dir_name:
                old_path = os.path.join(root, dir_name)
                new_dir_name = dir_name.replace(query_string, target_string)
                new_path = os.path.join(root, new_dir_name)
                
                if dry_run:
                    print(f"📋 [DRY RUN] 변경 예정:")
                    print(f"   이전: {old_path}")
                    print(f"   이후: {new_path}")
                else:
                    try:
                        os.rename(old_path, new_path)
                        print(f"✅ 디렉터리 이름 변경 완료:")
                        print(f"   이전: {old_path}")
                        print(f"   이후: {new_path}")
                        renamed_count += 1
                    except OSError as e:
                        print(f"❌ 디렉터리 이름 변경 실패: {old_path}")
                        print(f"   오류: {e}")
    
    if dry_run:
        print(f"🔍 총 {renamed_count}개의 디렉터리가 변경 대상입니다.")
        print("실제로 변경하려면 --dry_run=False 옵션을 사용하세요.")
    else:
        print(f"✅ 총 {renamed_count}개의 디렉터리 이름을 변경했습니다.")
    
    return renamed_count

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Aggregate results from res.json files and rename directories.')
    parser.add_argument('--dir_path', type=str, default="warehouse", help='Directory path to search for res.json files')
    parser.add_argument('--query_string', type=str, default='num_steps=7-unknown--network.num_steps=7', help='String to search for in directory names')
    parser.add_argument('--target_string', type=str, default='num_steps=4', help='String to replace query_string with')
    parser.add_argument('--dry_run', action='store_false', default=True, help='If True, only show what would be renamed without actually renaming')
    parser.add_argument('--rename_only', action='store_true', help='Only rename directories, skip result aggregation')
    args = parser.parse_args()

    print(f"🚀 작업 시작:")
    print(f"   디렉터리 경로: {args.dir_path}")
    print(f"   검색 문자열: {args.query_string}")
    print(f"   교체 문자열: {args.target_string}")
    print(f"   드라이런 모드: {args.dry_run}")
    print(f"   이름 변경만: {args.rename_only}")
    print()

    if args.rename_only:
        # 디렉터리 이름 변경만 수행
        rename_directories_with_query_string(
            args.dir_path, 
            args.query_string, 
            args.target_string, 
            args.dry_run
        )
    else:
        # 먼저 디렉터리 이름 변경
        print("🔄 Step 1: 디렉터리 이름 변경")
        rename_directories_with_query_string(
            args.dir_path, 
            args.query_string, 
            args.target_string, 
            args.dry_run
        )
        
        print("\n🔄 Step 2: 결과 집계")
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
                f_baseline = tokens[2]
                filename = os.path.basename(os.path.dirname(os.path.dirname(res_file)))

                # filename에 query_string이 포함된 경우, 그것을 target_string으로 변경 (이미 디렉터리는 변경되었으므로 이 부분은 필요없을 수 있음)
                if args.query_string in filename:
                    filename = filename.replace(args.query_string, args.target_string)

                results.append({
                    'source': source,
                    'baseline': f_baseline,
                    'filename': filename,
                    'r2': test_r2,
                    'rse': test_rse,
                    'file_path': res_file
                })

        print(f"📊 총 {len(results)}개의 결과 파일을 찾았습니다.")
        
        # CSV 파일로 저장 (선택사항)
        # import pandas as pd
        # df = pd.DataFrame(results)
        # df.to_csv('aggregated_results.csv', index=False)
        # print("📄 결과를 'aggregated_results.csv' 파일로 저장했습니다.")