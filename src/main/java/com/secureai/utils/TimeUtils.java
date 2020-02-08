package com.secureai.utils;

public class TimeUtils {
    public static long start = System.currentTimeMillis();

    public static long getStartMillis() {
        return start;
    }

    public static void setupStartMillis() {
        start = System.currentTimeMillis();
    }
}
