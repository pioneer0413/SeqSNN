## 1. 실험 수행
실행 전 `source`를 반드시 로컬 환경에 맞게 설정
### Non-spiking(ANN-based)
```
python scripts/run_experiments_nonspiking.py
```
### Baseline
```
python scripts/run_experiments.py
```
### Ours
```
python scripts/run_experiments.py --use_cluster
```

## 2. 결과 확보
#### 파일 설명
- `data_manager.py`: 설정 키 값을 참조하여, CSV 형식으로 결과 파일 생성
    - `target_dir`을 반드시 로컬 환경에 맞게 설정
- `discard_invalid.py`: 중단되거나, 결과가 정상적으로 존재하지 않는 디렉터리를 삭제
- `get_headers.py`: 저장되어 있는 각 결과 디렉터리로부터 설정 키 값을 추출
#### 사용 방법
1. 오류로 결과가 존재하지 않는 비정상 디렉터리 삭제
    ```
    python scripts/discard_invalid.py
    ```
2. 결과 디렉터리를 순회하며, 모든 테스트 케이스의 설정 키 값을 확보
    ```
    python scripts/get_headers.py
    ```
3. STDOUT에 출력된 문자열 설정 키 값을 복사하여 `raw_str`에 붙여넣기 후 아래 실행
    ```
    python scripts/data_manager.py
    ```
4. CSV 결과 파일 획득 및 후속 분석 작업에 활용