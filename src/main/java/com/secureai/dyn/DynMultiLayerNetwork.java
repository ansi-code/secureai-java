package com.secureai.dyn;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DynMultiLayerNetwork extends FilteredMultiLayerNetwork {
    public DynMultiLayerNetwork(MultiLayerConfiguration conf) {
        super(conf);
    }

    public DynMultiLayerNetwork(String conf, INDArray params) {
        super(conf, params);
    }

    public DynMultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
        super(conf, params);
    }
}
