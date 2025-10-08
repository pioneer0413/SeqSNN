import SeqSNN.module.encoding.snntorch as snn
import SeqSNN.module.encoding.spikingjelly as sj
import SeqSNN.module.encoding.spikingjelly.encoder_extend as sj_ext

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
        #"cwconv": sj.encoder.Cluster_wise_ConvEncoder,
        "cwconv": sj_ext.Cluster_wise_ConvEncoder,
    },
}
