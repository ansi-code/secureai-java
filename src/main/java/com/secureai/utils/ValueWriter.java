package com.secureai.utils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class ValueWriter {
    public static <T> void writeValue(String path, T value) {
        try {
            File f = new File(path);
            if (f.exists()) f.delete();
            f.getParentFile().mkdirs();
            BufferedWriter writer = new BufferedWriter(new FileWriter(path));
            writer.write(value.toString());
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
