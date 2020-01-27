package com.secureai.utils;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;

public class RLStatTrainingListener implements TrainingListener {

    private Stat<Double> stat;

    public RLStatTrainingListener(String path) {
        this.stat = new Stat<>(path + "/" + "reward.csv");
    }

    @Override
    public ListenerResponse onTrainingStart() {
        return null;
    }

    @Override
    public void onTrainingEnd() {

    }

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer iEpochTrainer) {
        return null;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, StatEntry statEntry) {
        this.stat.append(statEntry.getReward());
        this.stat.flush();
        return null;
    }

    @Override
    public ListenerResponse onTrainingProgress(ILearning iLearning) {
        return null;
    }
}
