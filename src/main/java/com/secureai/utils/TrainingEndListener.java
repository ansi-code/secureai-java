package com.secureai.utils;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.util.IDataManager;

public abstract class TrainingEndListener implements TrainingListener {
    @Override
    public ListenerResponse onTrainingStart() {
        return null;
    }

    @Override
    public abstract void onTrainingEnd();

    @Override
    public ListenerResponse onNewEpoch(IEpochTrainer iEpochTrainer) {
        return null;
    }

    @Override
    public ListenerResponse onEpochTrainingResult(IEpochTrainer iEpochTrainer, IDataManager.StatEntry statEntry) {
        return null;
    }

    @Override
    public ListenerResponse onTrainingProgress(ILearning iLearning) {
        return null;
    }
}
