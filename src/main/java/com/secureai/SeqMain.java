package com.secureai;

public class SeqMain {
    public static void main(String[] args) throws Exception {
        // THIS IS THE TEST FOR LAYERS COMPARISON
        DQNMain.main("--layers", "1", "--topology", "0");
        DQNMain.main("--layers", "2", "--topology", "0");
        DQNMain.main("--layers", "3", "--topology", "0");
        DQNMain.main("--layers", "4", "--topology", "0");

        DQNMain.main("--layers", "1", "--topology", "paper-4");
        DQNMain.main("--layers", "2", "--topology", "paper-4");
        DQNMain.main("--layers", "3", "--topology", "paper-4");
        DQNMain.main("--layers", "4", "--topology", "paper-4");

        // THIS IS THE TEST FOR DQN GAMMA
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--gamma", "0.25");
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--gamma", "0.5");
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--gamma", "0.75");
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--gamma", "1");

        // THIS IS THE TEST FOR DDQN
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--gamma", "0.75", "--doubleDQN", "true");

        // THIS IS THE TEST FOR STANDARD/PARALLEL/SPARK
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--dqn", "standard");
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--dqn", "parallel");
        DQNMain.main("--layers", "3", "--topology", "paper-4", "--dqn", "spark");

        // THIS IS THE TEST FOR VI/QN/DQN
        QNMain.main("--topology", "paper-4");
        DQNMain.main("--layers", "3", "--topology", "paper-4");
        VIMain.main("--topology", "paper-4");

        // THIS IS THE TEST FOR DynDDQN
        DynDQNMain.main("--layers", "3", "--gamma", "0.75");
        DynDQNMain.main("--layers", "3", "--gamma", "0.75", "--doubleDQN", "true");
    }
}
