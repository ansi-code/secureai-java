package com.secureai.utils;

public class NumberUtils {
    public static boolean hasValue(Double number) {
        return !(number.isInfinite() || number.isNaN());
    }
}
