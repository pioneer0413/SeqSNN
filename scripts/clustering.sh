'''
This is just for the testing purpose.
'''
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_cluster_electricity.yml --runner.max_epoches=3 --network.encoder_type=repeat --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_clustering