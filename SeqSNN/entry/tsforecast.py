'''
Module: tsforecast.py
Modified by: Hyunwoo Kang
Last Modified: 2025-10-08 16:51
Changes: 과도하게 용량이 큰 파일 생성을 방지하기 위해 46:48 라인 주석 처리
'''
import warnings

from utilsd import get_output_dir, get_checkpoint_dir, setup_experiment
from utilsd.experiment import print_config
from utilsd.config import PythonConfig, RegistryConfig, RuntimeConfig, configclass

from SeqSNN.dataset import DATASETS
from SeqSNN.runner import RUNNERS
from SeqSNN.network import NETWORKS

import time

warnings.filterwarnings("ignore")


@configclass
class SeqSNNConfig(PythonConfig):
    data: RegistryConfig[DATASETS]
    network: RegistryConfig[NETWORKS]
    runner: RegistryConfig[RUNNERS]
    runtime: RuntimeConfig = RuntimeConfig()


def run_train(config):
    setup_experiment(config.runtime)
    print_config(config)
    trainset = config.data.build(dataset_name="train")
    validset = config.data.build(dataset_name="valid")
    testset = config.data.build(dataset_name="test")
    network = config.network.build(
        input_size=trainset.num_variables, max_length=trainset.max_seq_len
    )
    runner = config.runner.build(
        network=network,
        output_dir=get_output_dir(),
        checkpoint_dir=get_checkpoint_dir(),
        out_size=config.runner.out_size or trainset.num_classes,
    )
    runner.fit(trainset, validset, testset)
    #runner.predict(trainset, "train") # Hyunwoo Kang에 의해 추가/수정되었음 (Research-Extended Version)
    #runner.predict(validset, "valid") # Hyunwoo Kang에 의해 추가/수정되었음 (Research-Extended Version)
    #runner.predict(testset, "test") # Hyunwoo Kang에 의해 추가/수정되었음 (Research-Extended Version)


if __name__ == "__main__":
    _config = SeqSNNConfig.fromcli()
    run_train(_config)