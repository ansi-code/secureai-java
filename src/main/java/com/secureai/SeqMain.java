package com.secureai;

public class SeqMain {
    public static void main(String[] args) throws Exception {
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
    }
}
