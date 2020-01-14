package com.secureai.utils;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.Arrays;

public class LoggingInvocationHandler implements InvocationHandler {
    private final Object delegate;

    public LoggingInvocationHandler(final Object delegate) {
        this.delegate = delegate;
    }

    @SuppressWarnings("unchecked")
    public static <T> T OfInstance(T instance, Class<T> interfaceClass) {
        return (T) Proxy.newProxyInstance(instance.getClass().getClassLoader(), new Class[]{interfaceClass}, new LoggingInvocationHandler(instance));
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        System.out.println("method: " + method + ", args: " + Arrays.toString(args));
        try {
            final Object ret = method.invoke(delegate, args);
            System.out.println("return: " + ret);
            return ret;
        } catch (Throwable t) {
            System.out.println("thrown: " + t);
            throw t;
        }
    }
}
