package com.secureai.utils;

import lombok.Getter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Stat<T> {
    @Getter
    private List<Timestamped<T>> history;
    @Getter
    private BufferedWriter bufferedWriter;
    private FileWriter fileWriter;

    public Stat() {
        this.history = new ArrayList<>();
    }

    public Stat(String filePath) {
        this();

        try {
            File f = new File(filePath);
            if (f.exists()) f.delete();
            f.getParentFile().mkdirs();

            this.fileWriter = new FileWriter(f);
            this.bufferedWriter = new BufferedWriter(this.fileWriter);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Stat<T> append(T value) {
        Timestamped<T> t = new Timestamped<>(value);
        this.history.add(t);

        if (this.bufferedWriter != null) {
            try {
                this.bufferedWriter.write(t.getTimestamp() + ", " + t.getValue() + "\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return this;
    }


    public void print() {
        System.out.println(this.toString());
    }

    public void write(String filePath) {
        try {
            Files.write(Paths.get(filePath), this.toString().getBytes());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void close() {
        if (this.bufferedWriter != null) {
            try {
                this.bufferedWriter.close();
                this.fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void flush() {
        if (this.bufferedWriter != null) {
            try {
                this.bufferedWriter.flush();
                this.fileWriter.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public String toString() {
        return history.stream().map(item -> item.getTimestamp() + ", " + item.getValue()).collect(Collectors.joining("\n"));
    }
}
