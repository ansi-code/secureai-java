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
        /*
        INDArray out = super.output(input, train, featuresMask, labelsMask);
        INDArray lbl = this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask;
        INDArray mskOut = super.output(input, train, featuresMask, lbl);
        for (int i = 0; i < input.rows(); i++) {
            System.out.println("out: " + out.getRow(i));
            System.out.println("lbl: " + lbl.getRow(i));
            System.out.println("mskOut: " + mskOut.getRow(i));
        }
        */
        return super.output(input, train, featuresMask, this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask);

        //return super.output(input, train, featuresMask, labelsMask);
    }

    public interface MultiLayerNetworkPredictionFilter {
        INDArray run(INDArray input);
    }
}
