package com.secureai.utils;

public class CallbackUtils {
    public interface NoArgsCallback {
        void callback();
    }

    public interface SingleArgCallback<A> {
        void callback(A a);
    }

    public interface DoubleArgsCallback<A, B> {
        void callback(A a, B b);
    }
}
