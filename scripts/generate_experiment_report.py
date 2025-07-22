#!/usr/bin/env python3
"""
SeqSNN 실험 결과 문서화 자동화 스크립트
outputs 폴더의 실험 결과를 읽어서 마크다운 표로 생성합니다.
"""

import os
import json
import re
from collections import defaultdict
from typing import Dict, Any, Optional

def count_earlystop_in_log(log_file_path: str) -> int:
    """
    stdout.log 파일에서 'Earlystop' 문자열의 개수를 세는 함수
    """
    if not os.path.exists(log_file_path):
        return 0
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count('Earlystop')
    except Exception as e:
        print(f"Warning: Error reading {log_file_path}: {e}")
        return 0

def read_json_metrics(json_file_path: str) -> Dict[str, Optional[float]]:
    """
    res.json 파일에서 R^2와 RSE 값을 읽는 함수
    """
    if not os.path.exists(json_file_path):
        return {'R^2': None, 'RSE': None}
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return {
                'R^2': data.get('test', {}).get('r2'),
                'RSE': data.get('test', {}).get('rse')
            }
    except Exception as e:
        print(f"Warning: Error reading {json_file_path}: {e}")
        return {'R^2': None, 'RSE': None}

def parse_folder_name(folder_name: str) -> Optional[Dict[str, str]]:
    """
    폴더명을 파싱하여 데이터세트명, 인코딩명, 베타수치를 추출하는 함수
    """
    pattern = r'spikernn_(.+)_encoder=(.+)_clustering_max_epoches=1000_wloss_vectorized_beta=(.+)'
    match = re.match(pattern, folder_name)
    
    if match:
        return {
            'dataset': match.group(1),
            'encoder': match.group(2),
            'beta': match.group(3)
        }
    return None

def collect_experiment_data(outputs_dir: str) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """
    outputs 폴더에서 실험 데이터를 수집하는 함수
    
    Returns:
        dict: {dataset: {beta: {encoder: {metric: value}}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    if not os.path.exists(outputs_dir):
        print(f"Error: {outputs_dir} 폴더가 존재하지 않습니다.")
        return {}
    
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # 폴더명 파싱
        parsed = parse_folder_name(folder_name)
        if not parsed:
            continue
        
        dataset = parsed['dataset']
        encoder = parsed['encoder']
        beta = parsed['beta']
        
        # stdout.log에서 termination 값 읽기
        log_file = os.path.join(folder_path, 'stdout.log')
        termination = count_earlystop_in_log(log_file)
        
        # res.json에서 메트릭 값 읽기
        json_file = os.path.join(folder_path, 'checkpoints', 'res.json')
        metrics = read_json_metrics(json_file)
        
        # 데이터 저장
        data[dataset][beta][encoder] = {
            'R^2': metrics['R^2'],
            'RSE': metrics['RSE'],
            'termination': termination
        }
        
        print(f"Processed: {dataset} - {encoder} - {beta}")
    
    return data

def calculate_rse_variance(data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], 
                          dataset: str, beta: str, encoder: str, baseline_data: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """
    baseline 대비 RSE의 변화율(%)을 계산하는 함수
    
    Args:
        data: 전체 데이터
        dataset: 데이터세트명
        beta: 베타값
        encoder: 인코더명
        baseline_data: baseline 데이터
    
    Returns:
        변화율(%) 또는 None (계산 불가시)
    """
    try:
        # baseline RSE 값
        baseline_rse = baseline_data[encoder].get('RSE')
        # 현재 RSE 값
        current_rse = data[dataset][beta][encoder].get('RSE')
        
        if baseline_rse is None or current_rse is None or baseline_rse == 0:
            return None
        
        # 변화율 계산: ((baseline - current) / baseline) * 100
        variance = ((baseline_rse - current_rse) / baseline_rse) * 100
        return variance
        
    except (KeyError, ZeroDivisionError):
        return None


def format_value(value: Any, metric_type: str = 'other') -> str:
    """
    값을 문자열로 포맷팅하는 함수
    
    Args:
        value: 포맷팅할 값
        metric_type: 메트릭 타입 ('R^2', 'RSE', 'Variance', 'other')
    
    Returns:
        포맷팅된 문자열
    """
    if value is None:
        return "N/A"
    elif isinstance(value, float) and metric_type in ['R^2', 'RSE', 'Variance']:
        # R^2, RSE, Variance(%)는 소수점 4번째 자리에서 반올림
        return f"{round(value, 4):.4f}"
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return str(value)

def generate_markdown_table(data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], 
                          encoders: list, datasets: list, betas: list) -> str:
    """
    수집된 데이터를 마크다운 표로 변환하는 함수
    """
    markdown = "# SeqSNN 실험 결과 보고서\n\n"
    
    for dataset in datasets:
        if dataset not in data:
            continue
            
        markdown += f"## 데이터세트: {dataset}\n\n"
        
        # baseline 데이터 가져오기 (variance 계산용)
        baseline_data = data[dataset].get('baseline', {})
        
        for beta in betas:
            if beta not in data[dataset]:
                continue
                
            markdown += f"### 베타수치: {beta}\n"
            markdown += "| 메트릭 | " + " | ".join(encoders) + " |\n"
            markdown += "|" + "--------|" * (len(encoders) + 1) + "\n"
            
            # baseline인 경우 Variance(%) 행 제외
            if beta == 'baseline':
                metrics = ['R^2', 'RSE', 'termination']
            else:
                metrics = ['R^2', 'RSE', 'Variance (%)', 'termination']
            
            # 각 메트릭별로 행 생성
            for metric in metrics:
                row = f"| {metric} |"
                for encoder in encoders:
                    if encoder in data[dataset][beta]:
                        if metric == 'Variance (%)':
                            # RSE variance 계산 (원본 값 사용)
                            variance = calculate_rse_variance(data, dataset, beta, encoder, baseline_data)
                            if variance is not None:
                                row += f" {format_value(variance, 'Variance')} |"
                            else:
                                row += " N/A |"
                        else:
                            value = data[dataset][beta][encoder].get(metric)
                            # R^2, RSE는 소수점 5번째 자리에서 반올림, termination은 그대로
                            if metric in ['R^2', 'RSE']:
                                row += f" {format_value(value, metric)} |"
                            else:
                                row += f" {format_value(value)} |"
                    else:
                        row += " N/A |"
                markdown += row + "\n"
            
            markdown += "\n"
        
        markdown += "---\n\n"
    
    return markdown

def generate_json_output(data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]], 
                        encoders: list, datasets: list, betas: list) -> Dict[str, Any]:
    """
    수집된 데이터를 JSON 형태로 변환하는 함수 (raw 값, 반올림 없음)
    
    Returns:
        JSON 형태의 딕셔너리
    """
    json_data = {
        "metadata": {
            "datasets": datasets,
            "encoders": encoders,
            "betas": betas,
            "generated_at": "2025-07-20"
        },
        "results": {}
    }
    
    for dataset in datasets:
        if dataset not in data:
            continue
            
        json_data["results"][dataset] = {}
        baseline_data = data[dataset].get('baseline', {})
        
        for beta in betas:
            if beta not in data[dataset]:
                continue
                
            json_data["results"][dataset][beta] = {}
            
            for encoder in encoders:
                if encoder in data[dataset][beta]:
                    encoder_data = data[dataset][beta][encoder]
                    
                    # 기본 메트릭
                    result = {
                        "R^2": encoder_data.get('R^2'),
                        "RSE": encoder_data.get('RSE'),
                        "termination": encoder_data.get('termination')
                    }
                    
                    # baseline이 아닌 경우 variance 계산 (raw 값)
                    if beta != 'baseline' and baseline_data and encoder in baseline_data:
                        variance = calculate_rse_variance(data, dataset, beta, encoder, baseline_data)
                        result["Variance(%)"] = variance
                    
                    json_data["results"][dataset][beta][encoder] = result
    
    return json_data

def main():
    """
    메인 함수
    """
    # 설정
    outputs_dir = './outputs'  # outputs 폴더 경로
    output_file = 'experiment_results.md'  # 마크다운 출력 파일명
    json_output_file = 'experiment_results.json'  # JSON 출력 파일명
    
    # 예상되는 값들 (실제 폴더에 따라 자동 감지됨)
    expected_datasets = ['metr-la', 'pems-bay', 'solar']
    expected_encoders = ['repeat', 'delta', 'conv']
    expected_betas = ['baseline', '1e-05', '1e-08', '1e-10']

    print("SeqSNN 실험 결과 문서화를 시작합니다...")
    print(f"outputs 폴더: {os.path.abspath(outputs_dir)}")
    
    # 데이터 수집
    data = collect_experiment_data(outputs_dir)
    
    if not data:
        print("수집된 데이터가 없습니다. outputs 폴더와 파일들을 확인해주세요.")
        return
    
    '''
    # 실제 존재하는 값들 추출
    actual_datasets = sorted(data.keys())
    actual_encoders = set()
    actual_betas = set()
    
    for dataset_data in data.values():
        for beta, beta_data in dataset_data.items():
            actual_betas.add(beta)
            for encoder in beta_data.keys():
                actual_encoders.add(encoder)
    
    actual_encoders = sorted(actual_encoders)
    actual_betas = sorted(actual_betas, key=lambda x: (x != 'baseline', x))  # baseline을 첫 번째로
    '''
    datasets = expected_datasets
    encoders = expected_encoders
    betas = expected_betas

    print(f"발견된 데이터세트: {datasets}")
    print(f"발견된 인코더: {encoders}")
    print(f"발견된 베타값: {betas}")

    # 마크다운 생성
    markdown_content = generate_markdown_table(data, encoders, datasets, betas)
    
    # JSON 생성 (raw 값)
    json_data = generate_json_output(data, encoders, datasets, betas)

    # 마크다운 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    # JSON 파일 저장
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n마크다운 결과가 {output_file}에 저장되었습니다!")
    print(f"JSON 결과가 {json_output_file}에 저장되었습니다!")
    
    # 요약 정보 출력
    total_experiments = sum(
        len(beta_data) for dataset_data in data.values() 
        for beta_data in dataset_data.values()
    )
    print(f"총 처리된 실험 수: {total_experiments}")

if __name__ == "__main__":
    main()
