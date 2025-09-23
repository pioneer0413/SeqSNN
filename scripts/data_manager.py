import pandas as pd
import numpy as np
import os
import json

if __name__ == '__main__':
    raw_str = 'data.dataset_name,data.file,data.horizon,data.last_label,data.normalize,data.raw_label,data.test_ratio,data.train_ratio,data.window,network.activation,network.cell_type,network.channel,network.class_strategy,network.common_thr,network.d_ff,network.d_model,network.depths,network.dilation,network.dim,network.dropout,network.e_layers,network.emb_dim,network.emb_type,network.embed,network.encoder_type,network.factor,network.freq,network.gpu_id,network.grad_slope,network.heads,network.hidden_size,network.injection_type,network.input_size,network.is_bidir,network.kernel_size,network.layers,network.max_length,network.n_cluster,network.n_heads,network.neuron_pe_scale,network.num_layers,network.num_levels,network.num_pe_neuron,network.num_steps,network.output_attention,network.pe_mode,network.pe_type,network.position_embedding,network.qk_scale,network.qkv_bias,network.stride,network.use_all_random,network.use_all_zero,network.use_cluster,network.use_ste,network.weight_file,runner.aggregate,runner.batch_size,runner.beta,runner.checkpoint_dir,runner.d_model,runner.denormalize,runner.early_stop,runner.loss_fn,runner.lower_is_better,runner.lr,runner.max_epoches,runner.metrics,runner.model_path,runner.n_cluster,runner.n_vars,runner.network,runner.observe,runner.optimizer,runner.out_ranges,runner.out_size,runner.output_dir,runner.seq_len,runner.task,runner.use_cluster,runner.valid_variates,runner.weight_decay,runtime.checkpoint_dir,runtime.debug,runtime.output_dir,runtime.seed,runtime.tb_log_dir,runtime.use_cuda,network.k_c,network.k_t'
    headers = raw_str.split(',')

    addtional_headers = [
        'target.architecture',
        'target.dataset',
        'target.encoder',
        'target.method',
        'target.postfix',
    ]

    headers.extend(addtional_headers)

    '''
    for header in headers:
        print(header)
    '''

    target_dir = 'warehouse/'

    def find_files(root: str, filename: str = "config.json"):
        root = os.path.abspath(os.path.expanduser(root))
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            if filename in filenames:
                found.append(os.path.join(dirpath, filename))
        # 중복 제거 및 경로 정렬
        return sorted(set(found))

    # target_dir 아래 모든 config.json 경로 리스트
    config_paths = find_files(target_dir, "config.json")

    print(f"found {len(config_paths)} config.json files under {os.path.abspath(target_dir)}")
    # 필요 시 경로 출력
    # for p in config_paths:
    #     print(p)

    # 각 config.json에서 키 값 추출: 없으면 'empty'
    def get_value_by_path(data: dict, key: str, missing='empty'):
        cur = data
        for part in key.split('.'):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return missing
        # 리스트/딕셔너리는 JSON 문자열로 직렬화해 단일 셀로 저장
        if isinstance(cur, (list, dict)):
            try:
                return json.dumps(cur, ensure_ascii=False)
            except Exception:
                return str(cur)
        return cur

    # values_matrix: [[value1, value2, ...], ...]
    values_matrix = []
    for cfg_path in config_paths:
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"[warn] skip {cfg_path}: {e}")
            cfg = {}

        row = [get_value_by_path(cfg, key, missing='empty') for key in headers]
        values_matrix.append(row)

    print(f"built values matrix: {len(values_matrix)} rows x {len(headers)} cols")

    # 필요 시 DataFrame으로 변환하여 확인/저장 가능
    df = pd.DataFrame(values_matrix, columns=headers)
    print(df.head())

    source_dirs = [os.path.dirname(p) for p in config_paths]
    
    for idx, source_dir in enumerate(source_dirs):
        dir_name = os.path.basename(source_dir)
        parts = dir_name.split('_')
        architecture = parts[0]
        dataset = parts[1]
        encoder = parts[2].split('=')[-1]
        method = os.path.dirname(source_dir).split('/')[-1]
        # 'p=' 다음으로 오는 부분이 postfix
        postfix = dir_name.split('p=')[-1] if 'p=' in dir_name else 'none'

        df.at[idx, 'target.architecture'] = architecture
        df.at[idx, 'target.dataset'] = dataset
        df.at[idx, 'target.encoder'] = encoder
        df.at[idx, 'target.method'] = method
        df.at[idx, 'target.postfix'] = postfix

        result_file = os.path.join(source_dir, 'checkpoints', 'res.json')
        # json파일에서 test.rrse의 값을 추출
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                res = json.load(f)
            rrse_value = get_value_by_path(res, 'test.rrse', missing='empty')
            rse_value = get_value_by_path(res, 'test.rse', missing='empty')
        except Exception as e:
            rrse_value = 'empty'
            rse_value = 'empty'
            #print(f"[warn] skip {result_file}: {e}")

        if rrse_value != 'empty':
            df.at[idx, 'target.rrse'] = rrse_value
        elif rse_value != 'empty':
            df.at[idx, 'target.rrse'] = rse_value
        else:
            df.at[idx, 'target.rrse'] = None  # rrse_value가 없으면 100.0으로 설정

    print(df.head())
    # 첫 번째 행의 모든 값 출력
    print("First row values with target:")
    for h in df.columns:
        print(f"{h}: {df.at[0, h]}")
    
    # df의 전체 행 수 출력
    print(f"DataFrame total rows: {len(df)}")

    output_file = 'outputs/results.csv'
    df.to_csv(output_file, index=False)
    print(f"saved results to {output_file}")