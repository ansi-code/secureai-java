package com.secureai.rl.vi;

import java.lang.reflect.Array;
import java.util.Arrays;

public class StaticIntegerMap<V extends Number> implements IntegerMap<V> {
    private V[] array;

    @SuppressWarnings("unchecked")
    public StaticIntegerMap(int size, Class<V> vClass) {
        this.array = (V[]) Array.newInstance(vClass, size);
    }

    @Override
    public V getOrDefault(int key, V value) {
        V res = this.array[key];
        return res != null ? res : value;
    }

    @Override
    public void put(int key, V value) {
        this.array[key] = value;
    }

    @Override
    public V get(int key) {
        return this.array[key];
    }

    @Override
    public String toString() {
        return "StaticIntegerMap{" +
                "array=" + Arrays.toString(array) +
                '}';
    }
}
