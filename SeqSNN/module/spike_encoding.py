'''
Module: spike_encoding.py
Modified by: Hyunwoo Kang
Last Modified: 2025-10-08 16:58
Changes: Cluster_wise_ConvEncoder 추가 및 SpikeEncoder 딕셔너리에 통합
'''
import SeqSNN.module.encoding.snntorch as snn
import SeqSNN.module.encoding.spikingjelly as sj
import SeqSNN.module.encoding.spikingjelly.encoder_extend as sj_ext # Hyunwoo Kang에 의해 추가/수정되었음 (Research-Extended Version)

SpikeEncoder = {
    "snntorch": {
        "repeat": snn.encoder.RepeatEncoder,
        "conv": snn.encoder.ConvEncoder,
        "delta": snn.encoder.DeltaEncoder,
    },
    "spikingjelly": {
        "repeat": sj.encoder.RepeatEncoder,
        "conv": sj.encoder.ConvEncoder,
        "delta": sj.encoder.DeltaEncoder,
        "cwconv": sj_ext.Cluster_wise_ConvEncoder, # Hyunwoo Kang에 의해 추가/수정되었음 (Research-Extended Version)
    },
}
