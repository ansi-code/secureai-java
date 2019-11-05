package com.secureai.utils;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.File;

public class YAMLParser {
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    public static <T> T parse(String filePath, Class<T> classType) {
        try {
            return yamlMapper.readValue(new File(filePath), classType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
