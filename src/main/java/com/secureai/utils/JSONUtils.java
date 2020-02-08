package com.secureai.utils;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

public class JSONUtils {

    private static ObjectMapper jsonMapper = new ObjectMapper(new YAMLFactory());

    public static <T> String toJSON(T t) {
        try {
            return jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(t);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return null;
    }
}
