for dataset in electricity metr-la pems-bay solar
do
    for encoder in repeat delta conv
    do
        for horizon in 6 24 48 96
        do

            python scripts/plot_prediction.py \
                --dataset $dataset \
                --compare \
                --compare_path_1 "/home/hwkang/SeqSNN/warehouse/with_pe/spikernn_${dataset}_encoder=${encoder}_horizon=${horizon}_baseline_seed=40" \
                --compare_path_2 "/home/hwkang/SeqSNN/warehouse/cluster/spikernn_cluster_${dataset}_encoder=${encoder}_horizon=${horizon}_baseline_seed=40" \

        done
    done
done
