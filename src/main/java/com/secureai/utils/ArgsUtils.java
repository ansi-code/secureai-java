package com.secureai.utils;

import java.util.HashMap;
import java.util.Map;

public class ArgsUtils {
    public static Map<String, String> toMap(String[] args) {
        Map<String, String> map = new HashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            map.put(args[i].substring(2), args[i + 1]);
        }
        return map;
    }
}
