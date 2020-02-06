package com.secureai.nn;

import com.secureai.utils.RandomUtils;
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
    public synchronized void fit(INDArray input, INDArray labels, INDArray featuresMask, INDArray labelsMask) {
        INDArray mask = this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask;
        super.fit(input, mask != null ? labels.muli(mask) : labels, featuresMask, labelsMask);
    }

    @Override
    public INDArray output(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask) {
        /*
        INDArray output = super.output(input, train, featuresMask, labelsMask);
        INDArray mask = this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask;
        INDArray maskOutput = output.muli(mask);
        //INDArray maskOutput = super.output(input, train, featuresMask, mask);
        for (int i = 0; i < input.rows(); i++) {
            System.out.println("input: " + input.getRow(i));
            System.out.println("output: " + output.getRow(i));
            System.out.println("mask: " + mask.getRow(i));
            System.out.println("maskOutput: " + maskOutput.getRow(i));
        }
        */


        //return super.output(input, train, featuresMask, this.multiLayerNetworkPredictionFilter != null ? this.multiLayerNetworkPredictionFilter.run(input) : labelsMask);
        //return super.output(input, train, featuresMask, labelsMask);
        //System.out.println(super.output(input, train, featuresMask, labelsMask));
        //System.out.println(this.multiLayerNetworkPredictionFilter != null ? super.output(input, train, featuresMask, labelsMask).muli(this.multiLayerNetworkPredictionFilter.run(input)) : super.output(input, train, featuresMask, labelsMask));
        //return this.multiLayerNetworkPredictionFilter != null ? super.output(input, train, featuresMask, labelsMask).muli(this.multiLayerNetworkPredictionFilter.run(input)) : super.output(input, train, featuresMask, labelsMask);
        INDArray res = this.multiLayerNetworkPredictionFilter != null ? super.output(input, train, featuresMask, labelsMask).muli(this.multiLayerNetworkPredictionFilter.run(input)) : super.output(input, train, featuresMask, labelsMask);
        for (int i = 0; i < res.rows(); i++) {
            if (res.getRow(i).maxNumber().equals(0d))
                res.put(i, RandomUtils.getRandom(0, res.columns() - 1), .5);
        }
        //System.out.println(res);
        return res;
    }

    public interface MultiLayerNetworkPredictionFilter {
        INDArray run(INDArray input);
    }
}
