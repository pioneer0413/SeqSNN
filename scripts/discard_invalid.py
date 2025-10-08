'''
Module: discard_invalid.py
Author: Hyunwoo Kang
Last Modified: 2025-10-08 17:13
Description: 특정 디렉터리를 순회하여 'checkpoints' 디렉터리가 존재하는 실험 중에서, 'checkpoints' 내부에 'res.json' 파일이 없는 경우 해당 실험 디렉터리를 삭제하는 스크립트
'''
target_dir = 'warehouse/'

import os
import shutil

def find_invalid_experiment_dirs(root: str, checkpoints_name: str = "checkpoints", result_file: str = "res.json"):
    root = os.path.abspath(os.path.expanduser(root))
    invalid = set()
    for dirpath, dirnames, filenames in os.walk(root):
        # checkpoints 디렉터리 자체에 도달했을 때만 검사
        if os.path.basename(dirpath) == checkpoints_name:
            parent = os.path.dirname(dirpath)
            if result_file not in filenames:
                invalid.add(parent)
            # checkpoints 내부는 더 순회할 필요 없음
            dirnames[:] = []
    return sorted(invalid)

if __name__ == "__main__":
    invalid_dirs = find_invalid_experiment_dirs(target_dir, "checkpoints", "res.json")
    root_abs = os.path.abspath(os.path.expanduser(target_dir))

    print(f"invalid experiment dirs (missing checkpoints/res.json): {len(invalid_dirs)}")
    '''for d in invalid_dirs:
        print(d)
    print(len(invalid_dirs))'''

    if invalid_dirs:
        ans = input("Delete ALL invalid directories above? Type 'yes' to confirm: ").strip().lower()
        if ans == "yes":
            deleted = 0
            for d in invalid_dirs:
                try:
                    d_abs = os.path.abspath(d)
                    if not d_abs.startswith(root_abs + os.sep):
                        print(f"[skip] outside root: {d_abs}")
                        continue
                    if os.path.isdir(d_abs):
                        shutil.rmtree(d_abs)
                        deleted += 1
                        print(f"[deleted] {d_abs}")
                    else:
                        print(f"[skip] not a directory: {d_abs}")
                except Exception as e:
                    print(f"[error] failed to delete {d}: {e}")
            print(f"deleted {deleted}/{len(invalid_dirs)} directories")
        else:
            print("deletion canceled")
    else:
        print("no invalid directories")