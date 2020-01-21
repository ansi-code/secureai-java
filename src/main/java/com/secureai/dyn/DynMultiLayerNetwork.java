package com.secureai.dyn;

import com.secureai.utils.BidirectionalMap;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

public class DynMultiLayerNetwork extends FilteredMultiLayerNetwork {

    @Getter @Setter
    private BidirectionalMap<?, Integer> inputMap;

    @Getter @Setter
    private BidirectionalMap<?, Integer> outputMap;

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
