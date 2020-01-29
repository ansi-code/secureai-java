package com.secureai.utils;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;

public abstract class EpochEndListener implements TrainingListener {
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
    public abstract ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, StatEntry statEntry);

    @Override
    public ListenerResponse onTrainingProgress(ILearning iLearning) {
        return null;
    }
}
