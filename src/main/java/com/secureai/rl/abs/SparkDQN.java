package com.secureai.rl.abs;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.rl4j.network.dqn.DQN;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class SparkDQN<NN extends DQN> extends DQN<NN> {
    JavaSparkContext spark;
    private SparkDl4jMultiLayer net;

    public SparkDQN(MultiLayerNetwork mln) {
        super(mln);
        SparkConf conf = new SparkConf().setAppName("SecureAI").setMaster("local");
        this.spark = new JavaSparkContext(conf);
        ParameterAveragingTrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(1).build();
        this.net = new SparkDl4jMultiLayer(this.spark, mln, trainingMaster);
    }

    @Override
    public void fit(INDArray input, INDArray labels) {
        this.net.fit(this.spark.parallelize(new DataSet(input, labels).asList()));
    }

    @Override
    public void fit(INDArray input, INDArray[] labels) {
        this.net.fit(this.spark.parallelize(new DataSet(input, Nd4j.vstack(labels)).asList()));
    }
}
