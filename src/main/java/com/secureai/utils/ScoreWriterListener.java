package com.secureai.utils;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;

import java.io.Serializable;

public class ScoreWriterListener extends BaseTrainingListener implements Serializable {
    private Stat<Double> stat;

    public ScoreWriterListener(String path) {
        this.stat = new Stat<>(path + "/" + "score.csv");
    }

    public void iterationDone(Model model, int iteration, int epoch) {
        stat.append(model.score());
        stat.flush();
    }
}
