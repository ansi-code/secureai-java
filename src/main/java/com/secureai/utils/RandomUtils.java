package com.secureai.utils;

import lombok.Getter;

import java.util.Random;

public class RandomUtils {
    @Getter
    public static Random random = new Random(12345);

    public static <T> T getRandom(T[] array) {
        int rnd = new Random().nextInt(array.length);
        return array[rnd];
    }
}
