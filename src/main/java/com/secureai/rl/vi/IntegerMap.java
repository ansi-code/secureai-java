package com.secureai.rl.vi;

public interface IntegerMap<V> {
    V getOrDefault(int key, V v);

    void put(int key, V v);

    V get(int key);
}
