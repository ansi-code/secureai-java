package com.secureai.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public class ArrayUtils {
    public static double max(double[] elems) {
        double max = Double.NEGATIVE_INFINITY;
        for (double elem : elems)
            max = Math.max(max, elem);
        return max;
    }

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

    public static long multiply(long[] a) {
        long result = 1;
        for (long v : a)
            result *= v;
        return result;
    }

    public static double[] multiply(double[] a, double[] b) {
        double[] result = new double[a.length];
        for (int i = 0; i < a.length; i++)
            result[i] = a[i] * b[i];
        return result;
    }

    public static int[] fromBinaryString(String binary) {
        int[] result = new int[binary.length()];
        for (int i = 0; i < binary.length(); i++) {
            result[i] = Integer.parseInt(String.valueOf(binary.charAt(i)));
        }
        return result;
    }

    public static String toIntString(int[] array) {
        char[] result = new char[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (char) (array[i] + '0');
        }
        return String.valueOf(result);
    }

    public static int toBase10(int[] values, int base) {
        int num = 0;
        for (int i = values.length - 1, power = 1; i >= 0; power *= base)
            num += values[i--] * power;
        return num;
    }

    public static int[] fromBase10(int value, int base) {
        List<Integer> result = new ArrayList<>();
        int num = value;
        while (num > 0) {
            int r = num % base;
            num /= base;
            result.add(0, r);
        }
        return result.size() == 0 ? new int[]{0} : org.apache.commons.lang3.ArrayUtils.toPrimitive(result.toArray(new Integer[0]));
    }
}
