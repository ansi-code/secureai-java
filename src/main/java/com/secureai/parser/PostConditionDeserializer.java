package com.secureai.parser;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.secureai.system.NodeActionDefinition;
import com.secureai.system.NodeState;
import com.secureai.utils.RandomUtils;
import org.apache.commons.lang.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PostConditionDeserializer extends StdDeserializer<NodeActionDefinition.PostNodeStateFunction> {

    public PostConditionDeserializer() {
        this(null);
    }

    public PostConditionDeserializer(Class<?> vc) {
        super(vc);
    }

    @Override
    public NodeActionDefinition.PostNodeStateFunction deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        return this.parsePostConditions(jsonParser.getValueAsString());
    }

    private NodeActionDefinition.PostNodeStateFunction parsePostConditions(String str) {
        if (str == null)
            return (state, i) -> {
            };

        List<NodeActionDefinition.PostNodeStateFunction> andConditions = new ArrayList<>();
        if (!str.contains(", "))
            andConditions.add(this.parsePostCondition(str));
        else {
            for (String andConditionString : str.split(", ")) {
                andConditions.add(this.parsePostCondition(andConditionString));
            }
        }

        return andConditions.stream().reduce((a, b) -> (state, i) -> {
            a.run(state, i);
            b.run(state, i);
        }).orElse(null);
    }

    private NodeActionDefinition.PostNodeStateFunction parsePostCondition(String str) {
        String[] components = str.split(" = ");
        NodeState nodeState = NodeState.valueOf(StringUtils.substringBetween(components[0], "[", "]"));
        double threshold = Double.valueOf(StringUtils.substringBetween(components[1], "rand(", ")"));

        return (state, i) -> state.set(i, nodeState, RandomUtils.random.nextDouble() < threshold);
    }
}
