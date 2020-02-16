package com.secureai.nn;

import com.secureai.utils.NumberUtils;
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
        //System.out.println(super.output(input, train, featuresMask, labelsMask));
        INDArray result = this.multiLayerNetworkPredictionFilter != null ? super.output(input, train, featuresMask, labelsMask).muli(this.multiLayerNetworkPredictionFilter.run(input)) : super.output(input, train, featuresMask, labelsMask);
        // This is needed to add some salt when we are masking too many actions
        for (int i = 0; i < result.rows(); i++)
            if (!NumberUtils.hasValue(result.getRow(i).maxNumber().doubleValue()))
                result.put(i, RandomUtils.getRandom(0, result.columns() - 1), 0);
        return result;
    }

    public interface MultiLayerNetworkPredictionFilter {
        INDArray run(INDArray input);
    }
}
