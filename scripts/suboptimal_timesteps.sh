'''
Sub-optimal timesteps
'''

# spikernn(T=2)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=2 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=2;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=2 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=2;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=2 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=2;

# spikernn(T=4)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=4 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=4;   \ 
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=4 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=4;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=4 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=4;

# spikernn(T=8)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=8 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=8;   \ 
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=8 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=8;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=8 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=8;    

# spikernn(T=16)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=16 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=16; \   
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=16 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=16;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=16 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=16;

# spikernn(T=32)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=32 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=32;   \ 
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=32 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=32;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=32 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=32;

# spikernn(T=64)
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=delta --network.num_steps=64 --runtime.output_dir=./outputs/spikernn_electricity_encoder=delta_T=64;   \ 
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=repeat --network.num_steps=64 --runtime.output_dir=./outputs/spikernn_electricity_encoder=repeat_T=64;\
python -m SeqSNN.entry.tsforecast exp/forecast/spikernn/spikernn_electricity.yml --network.encoder_type=conv --network.num_steps=64 --runtime.output_dir=./outputs/spikernn_electricity_encoder=conv_T=64;