package com.secureai;

import com.secureai.model.Topology;
import com.secureai.utils.YAML;

public class Main {

    public static void main(String... args) {
        System.out.println("Hello World");
        System.out.println(YAML.parse("data/topology-0.yml", Topology.class));
    }
}
