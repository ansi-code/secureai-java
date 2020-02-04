package com.secureai;

public class SequentialTest {
    public static void main(String[] args) throws Exception {
        DQNMain.main("--layers", "4");
        DQNMain.main("--layers", "2");
        DQNMain.main("--layers", "3");
        DQNMain.main("--layers", "4");
    }
}
