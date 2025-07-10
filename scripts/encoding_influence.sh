# spikernn
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=all0s --runtime.output_dir=./outputs/spikernn_electricity_encoder=all0s

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=all1s --runtime.output_dir=./outputs/spikernn_electricity_encoder=all1s

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=random --runtime.output_dir=./outputs/spikernn_electricity_encoder=random

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=delta --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=repeat --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat

python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --runner.max_epoches=5 --network.encoder_type=conv --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv

# ispikformer
python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=all0s --runtime.output_dir=./outputs/ispikformer_electricity_encoder=all0s

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=all1s --runtime.output_dir=./outputs/ispikformer_electricity_encoder=all1s

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=random --runtime.output_dir=./outputs/ispikformer_electricity_encoder=random

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=delta --runtime.output_dir=./outputs/ispikformer_electricity_encoder=delta

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=repeat --runtime.output_dir=./outputs/ispikformer_electricity_encoder=repeat

python -m SeqSNN.entry.tsforecast exp/forecast/ispikformer/ispikformer_electricity.yml --runner.max_epoches=5 --network.encoder_type=conv --runtime.output_dir=./outputs/ispikformer_electricity_encoder=conv