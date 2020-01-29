package com.secureai.model.stateset;

import lombok.Getter;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public enum State {

    active(0),
    updated(1),
    upgradable(2),
    corrupted(3),
    vulnerable(4);

    @Getter
    @NonNull
    private int value;

}
