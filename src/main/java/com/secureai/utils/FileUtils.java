package com.secureai.utils;

import java.io.File;

public class FileUtils {

    public static String firstAvailableFolder(String folder, String prefix) {
        for (int i = 0; ; i++) {
            String path = String.format("%s/%s-%d", folder, prefix, i);
            if (!new File(path).exists())
                return path;
        }
    }
}
