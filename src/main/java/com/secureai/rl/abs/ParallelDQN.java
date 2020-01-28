package com.secureai.rl.abs;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class ParallelDQN<NN extends DQN> extends DQN<NN> {
    private ParallelWrapper net;

    public ParallelDQN(MultiLayerNetwork mln) {
        super(mln);
        this.net = new ParallelWrapper.Builder<>(mln)
                .prefetchBuffer(24) // DataSets pre-fetching options. Set this value with respect to number of actual devices
                .workers(2) // set number of workers equal to number of available devices. x1-x2 are good values to start with
                .averagingFrequency(3) // rare averaging improves performance, but might reduce model accuracy
                .reportScoreAfterAveraging(true) // if set to TRUE, on every averaging model score will be reported
                .workspaceMode(WorkspaceMode.ENABLED) // 2 options here: NONE, ENABLED
                .build();
    }

    @Override
    public void fit(INDArray input, INDArray labels) {
        this.net.fit(new ListDataSetIterator<>(new DataSet(input, labels).asList()));
    }

    @Override
    public void fit(INDArray input, INDArray[] labels) {
        this.net.fit(new ListDataSetIterator<>(new DataSet(input, Nd4j.vstack(labels)).asList()));
    }
}
