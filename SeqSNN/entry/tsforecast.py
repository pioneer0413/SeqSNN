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
    runner.predict(trainset, "train")
    runner.predict(validset, "valid")
    runner.predict(testset, "test")


if __name__ == "__main__":
    _config = SeqSNNConfig.fromcli()

    time_start = time.time()
    run_train(_config)
    execution_time = time.time() - time_start

    save_path = 'outputs/execution_time.csv'
    header = ['network_type', 'encoder_type', 'num_steps', 'execution_time']
    record = [_config.runtime.output_dir, _config.network.encoder_type, _config.network.num_steps, execution_time]
    '''
    if not save_path exists, then create it and write the header with CSV format
    else append the record to the CSV file
    '''
    import os
    import csv
    if not os.path.exists(save_path):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(save_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(record)
