package com.secureai.utils;

import java.util.HashMap;
import java.util.Map;

public class MapCounter<T> {
    private Map<T, Integer> map;

    public MapCounter() {
        this.map = new HashMap<>();
    }

    public void increment(T key) {
        this.map.put(key, this.map.getOrDefault(key, 0) + 1);
    }

    public void decrement(T key) {
        this.map.put(key, this.map.getOrDefault(key, 0) - 1);
    }

    @Override
    public String toString() {
        return this.map.toString();
    }
}
