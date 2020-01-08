package com.secureai.utils;

public class Decorated<T> {
    private T value;

    public Decorated(T value) {
        this.value = value;
    }

    public T get() {
        return this.value;
    }
}
