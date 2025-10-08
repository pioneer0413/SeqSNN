'''
Module: get_headers.py
Author: Hyunwoo Kang
Last Modified: 2025-10-08 17:14
Description: 특정 디렉터리를 순회하여 모든 config.json 파일을 찾고, 각 파일에서 모든 키를 추출하여 CSV 헤더로 출력하는 스크립트
'''
import pandas as pd
import numpy as np
import os
import json
import sys
from typing import Any, Set

if __name__ == '__main__':
    root_dir = 'warehouse/'
    target_file = 'config.json'

    def find_files(root: str, filename: str):
        root = os.path.abspath(os.path.expanduser(root))
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                found.append(os.path.join(dirpath, filename))
        # 중복 제거 및 정렬
        return sorted(set(found))

    # 평탄화: dict는 하위로 내려가고, list/원시값은 현재 경로를 키로 등록
    def flatten_keys(obj: Any, parent: str = "") -> Set[str]:
        keys: Set[str] = set()
        if isinstance(obj, dict):
            if not obj and parent:
                keys.add(parent)
            for k, v in obj.items():
                prefix = f"{parent}.{k}" if parent else k
                keys |= flatten_keys(v, prefix)
        elif isinstance(obj, list):
            if parent:
                keys.add(parent)
        else:
            if parent:
                keys.add(parent)
        return keys

    def collect_keys_from_file(path: str) -> Set[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return flatten_keys(data)
        except Exception as e:
            print(f"[warn] skip {path}: {e}", file=sys.stderr)
            return set()

    def sort_headers(headers: Set[str]) -> list:
        section_order = {"data": 0, "network": 1, "runner": 2, "runtime": 3}
        return sorted(headers, key=lambda k: (section_order.get(k.split(".", 1)[0], 99), k))

    config_paths = find_files(root_dir, target_file)

    # 필요 시 출력
    '''
    print(f"found {len(config_paths)} files")
    for p in config_paths:
        print(p)
    '''

    # 모든 파일에서 키 수집
    all_keys: Set[str] = set()
    for p in config_paths:
        all_keys |= collect_keys_from_file(p)

    headers = sort_headers(all_keys)
    print(f"unique headers: {len(headers)}")
    # CSV 헤더 한 줄 출력
    print(",".join(headers))



