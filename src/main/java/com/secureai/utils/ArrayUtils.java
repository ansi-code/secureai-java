package com.secureai.utils;

import java.util.Arrays;
import java.util.stream.Stream;

public class ArrayUtils {
    public static int argMax(double[] elems) {
        int bestIdx = -1;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < elems.length; i++) {
            double elem = elems[i];
            if (elem > max) {
                max = elem;
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    public static <T> Stream<T> flatten(T[] array) {
        return Arrays.stream(array).flatMap(o -> array.getClass().isArray() ? flatten(array) : Stream.of(o));
    }

    public static double[] multiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++)
            result[i] = a[i] * b[i];
        return result;
    }
}
