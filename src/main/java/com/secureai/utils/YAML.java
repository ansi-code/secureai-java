package com.secureai.utils;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

import java.io.File;

public class YAML {
    private static ObjectMapper yamlMapper = new ObjectMapper(new YAMLFactory());

    public static <T> T parse(String filePath, Class<T> classType) {
        yamlMapper.setSerializationInclusion(JsonInclude.Include.NON_NULL);
        try {
            return yamlMapper.readValue(new File(filePath), classType);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
