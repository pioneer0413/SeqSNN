'''
Channel Shuffle
'''

# spikernn
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=1000 --network.encoder_type=delta --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=1000 --network.encoder_type=repeat --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=1000 --network.encoder_type=conv --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_channel_shuffle;

# ispikformer
python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=1000 --network.encoder_type=delta --runtime.output_dir=./outputs/ispikformer_electricity_encoder=delta_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=1000 --network.encoder_type=repeat --runtime.output_dir=./outputs/ispikformer_electricity_encoder=repeat_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=1000 --network.encoder_type=conv --runtime.output_dir=./outputs/ispikformer_electricity_encoder=conv_channel_shuffle;

# spiketcn
python -m SeqSNN.entry.tsforecast exp/forecast/tcn/spiketcn2d_electricity.yml --runner.max_epoches=1000 --network.encoder_type=delta --runtime.output_dir=./outputs/spiketcn_electricity_encoder=delta_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/tcn/spiketcn2d_electricity.yml --runner.max_epoches=1000 --network.encoder_type=repeat --runtime.output_dir=./outputs/spiketcn_electricity_encoder=repeat_channel_shuffle;

python -m SeqSNN.entry.tsforecast exp/forecast/tcn/spiketcn2d_electricity.yml --runner.max_epoches=1000 --network.encoder_type=conv --runtime.output_dir=./outputs/spiketcn_electricity_encoder=conv_channel_shuffle;