package com.secureai;

import com.secureai.model.actionset.ActionSet;
import com.secureai.utils.YAML;

public class DefinitionTest {

    public static void main(String[] args) throws Exception {
        ActionSet actionSet = YAML.parse("data/definition-2.yml", ActionSet.class);
        System.out.println(actionSet);
    }
}
