package com.secureai;

import com.fasterxml.jackson.databind.PropertyNamingStrategy;
import com.secureai.parser.Definition;
import com.secureai.utils.YAML;

public class DefinitionTest {

    public static void main(String[] args) throws Exception {
        Definition definition = YAML.parse("data/definition-2.yml", Definition.class);
        System.out.println(definition);
    }
}
