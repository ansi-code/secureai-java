package com.secureai.nn;

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
    public INDArray output(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask) {
        return super.output(input, train, featuresMask, this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask);
    }

    public interface MultiLayerNetworkPredictionFilter {
        INDArray run(INDArray input);
    }
}
