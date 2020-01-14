package com.secureai.utils;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

public class BidirectionalMap<K, V> extends HashMap<K, V> {
    private static final long serialVersionUID = 1L;

    private HashMap<V, K> inverseMap;

    public BidirectionalMap(Map<? extends K, ? extends V> m) {
        super(m);
        m.forEach((key, value) -> this.inverseMap.put(value, key));
    }

    public BidirectionalMap() {
        super();
        this.inverseMap = new HashMap<>();
    }

    @Override
    public int size() {
        return this.size();
    }

    @Override
    public boolean isEmpty() {
        return this.size() > 0;
    }

    @Override
    public V remove(Object key) {
        V value = super.remove(key);
        this.inverseMap.remove(value);
        return value;
    }

    @Override
    public V get(Object key) {
        return super.get(key);
    }

    @Override
    public V put(K key, V value) {
        this.inverseMap.put(value, key);
        return super.put(key, value);
    }

    @Override
    public void putAll(Map<? extends K, ? extends V> m) {
        m.forEach((key, value) -> this.inverseMap.put(value, key));
        super.putAll(m);
    }

    @Override
    public V putIfAbsent(K key, V value) {
        this.inverseMap.putIfAbsent(value, key);
        return super.putIfAbsent(key, value);
    }

    @Override
    public void replaceAll(BiFunction<? super K, ? super V, ? extends V> function) {
        super.replaceAll(function);
    }

    public HashMap<V, K> inverse() {
        return this.inverseMap;
    }

    public K getKey(V value) {
        return this.inverseMap.get(value);
    }

    @Override
    public V merge(K key, V value, BiFunction<? super V, ? super V, ? extends V> remappingFunction) {
        this.inverseMap.merge(value, key, (u, v) -> {
            throw new IllegalStateException(String.format("Duplicate key %s", u));
        });
        return super.merge(key, value, remappingFunction);
    }
}
