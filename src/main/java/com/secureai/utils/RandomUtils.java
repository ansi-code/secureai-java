package com.secureai.utils;

import lombok.Getter;

import java.util.Random;

public class RandomUtils {
    @Getter
    public static Random random = new Random(12345);
}
