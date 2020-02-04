package com.secureai.utils;

import lombok.Getter;

public class Timestamped<T> {
    @Getter
    private long timestamp;
    @Getter
    private T value;

    public Timestamped(T value) {
        this.timestamp = System.currentTimeMillis();
        this.value = value;
    }
}
