package com.secureai.utils;

import lombok.Getter;

import java.util.Random;

public class RandomUtils {
    @Getter
    public static Random random = new Random(12345);

    public static <T> T getRandom(T[] array) {
        int rnd = random.nextInt(array.length);
        return array[rnd];
    }

    public static int getRandom(int min, int max) {
        return random.nextInt((max - min) + 1) + min;
    }
}
