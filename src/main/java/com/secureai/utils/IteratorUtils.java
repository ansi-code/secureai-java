package com.secureai.utils;

import java.util.AbstractMap;
import java.util.Iterator;
import java.util.Map;

public class IteratorUtils {
    public static <T> T getAtIndex(Iterator<T> iterator, int i) {
        for (int j = 0; iterator.hasNext(); j++) {
            T v = iterator.next();
            if (i == j)
                return v;
        }
        return null;
    }

    public static <T> Iterator<Map.Entry<Integer, T>> zipWithIndex(Iterator<T> iterator) {
        return new Iterator<Map.Entry<Integer, T>>() {
            private int index = 0;

            @Override
            public boolean hasNext() {
                return iterator.hasNext();
            }

            @Override
            public Map.Entry<Integer, T> next() {
                return new AbstractMap.SimpleImmutableEntry<>(index++, iterator.next());
            }
        };
    }
}
