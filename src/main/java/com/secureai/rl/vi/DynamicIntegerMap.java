package com.secureai.rl.vi;

import com.secureai.utils.JSONUtils;

import java.util.HashMap;
import java.util.Map;

public class DynamicIntegerMap<V extends Number> implements IntegerMap<V> {
    private Map<Integer, V> map;

    public DynamicIntegerMap() {
        this.map = new HashMap<>();
    }

    @Override
    public V getOrDefault(int key, V value) {
        return this.map.getOrDefault(key, value);
    }

    @Override
    public void put(int key, V value) {
        this.map.put(key, value);
    }

    @Override
    public V get(int key) {
        return this.map.get(key);
    }

    @Override
    public String toString() {
        return "DynamicIntegerMap{" +
                "map=" + JSONUtils.toJSON(this.map) +
                '}';
    }
}
