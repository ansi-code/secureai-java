package com.secureai.utils;

import lombok.Getter;

import java.time.Instant;

public class Timestamped<T> {
    @Getter
    private Instant timestamp;
    @Getter
    private T value;

    public Timestamped(T value) {
        this.timestamp = Instant.now();
        this.value = value;
    }
}
