package com.secureai.dyn;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

public class FilteredMultiLayerNetwork extends MultiLayerNetwork {

    @Getter
    @Setter
    private MultiLayerNetworkPredictionFilter multiLayerNetworkPredictionFilter;

    public FilteredMultiLayerNetwork(MultiLayerConfiguration conf) {
        super(conf);
    }

    public FilteredMultiLayerNetwork(String conf, INDArray params) {
        super(conf, params);
    }

    public FilteredMultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
        super(conf, params);
    }

    @Override
    public int[] predict(INDArray d) {
        //TODO: Wrong, it should use this or this.output
        //this.setLayerMaskArrays(null, labelMask)
        return this.multiLayerNetworkPredictionFilter.run(d, super.predict(d));
    }

    public interface MultiLayerNetworkPredictionFilter {
        int[] run(INDArray input, int[] output);
    }
}
