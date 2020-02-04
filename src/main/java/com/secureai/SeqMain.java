package com.secureai;

public class SeqMain {
    public static void main(String[] args) throws Exception {
        /*
        // THIS IS THE TEST FOR CPU/GPU and layers comparison
        DQNMain.main("--layers", "1", "--topolgy", "0");
        DQNMain.main("--layers", "2", "--topolgy", "0");
        DQNMain.main("--layers", "3", "--topolgy", "0");
        DQNMain.main("--layers", "4", "--topolgy", "0");

        DQNMain.main("--layers", "1", "--topolgy", "paper-4");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4");
        DQNMain.main("--layers", "3", "--topolgy", "paper-4");
        DQNMain.main("--layers", "4", "--topolgy", "paper-4");

        DQNMain.main("--layers", "1", "--topolgy", "3");
        DQNMain.main("--layers", "2", "--topolgy", "3");
        DQNMain.main("--layers", "3", "--topolgy", "3");
        DQNMain.main("--layers", "4", "--topolgy", "3");
         */

        // THIS IS THE TEST FOR DQN GAMMA
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--gamma", "5");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--gamma", "10");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--gamma", "20");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--gamma", "40");

        // THIS IS THE TEST FOR DDQN
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--gamma", "5", "--doubleDQN", "true");

        // THIS IS THE TEST FOR VI/QN/DQN
        VIMain.main("--topolgy", "paper-4");
        QNMain.main("--topolgy", "paper-4");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4");

        // THIS IS THE TEST FOR STANDARD/PARALLEL/SPARK
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--dqn", "standard");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--dqn", "parallel");
        DQNMain.main("--layers", "2", "--topolgy", "paper-4", "--dqn", "spark");

        // THIS IS THE TEST FOR DynDDQN
        DQNMain.main("--layers", "2", "--gamma", "5");
    }
}
