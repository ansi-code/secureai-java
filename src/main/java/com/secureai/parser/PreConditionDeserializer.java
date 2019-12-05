package com.secureai.parser;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.secureai.system.NodeActionDefinition;
import com.secureai.system.NodeState;
import org.apache.commons.lang.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PreConditionDeserializer extends StdDeserializer<NodeActionDefinition.PreNodeStateFunction> {

    public PreConditionDeserializer() {
        this(null);
    }

    public PreConditionDeserializer(Class<?> vc) {
        super(vc);
    }

    @Override
    public NodeActionDefinition.PreNodeStateFunction deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException, JsonProcessingException {
        return this.parsePreConditions(jsonParser.getValueAsString());
    }

    private NodeActionDefinition.PreNodeStateFunction parsePreConditions(String str) {
        if (str == null)
            return (state, i) -> true;

        List<NodeActionDefinition.PreNodeStateFunction> andConditions = new ArrayList<>();
        if (!str.contains(" && "))
            andConditions.add(this.parsePreCondition(str));
        else {
            for (String andConditionString : str.split(" && ")) {
                List<NodeActionDefinition.PreNodeStateFunction> orConditions = new ArrayList<>();
                if (!str.contains(" \\|\\| "))
                    orConditions.add(this.parsePreCondition(andConditionString));
                else
                    for (String orConditionString : andConditionString.split(" \\|\\| "))
                        orConditions.add(this.parsePreCondition(orConditionString));
                andConditions.add(orConditions.stream().reduce((a, b) -> (state, i) -> a.run(state, i) || b.run(state, i)).orElse(null));
            }
        }

        return andConditions.stream().reduce((a, b) -> (state, i) -> a.run(state, i) && b.run(state, i)).orElse(null);
    }

    private NodeActionDefinition.PreNodeStateFunction parsePreCondition(String str) {
        String[] components = str.split(" == ");
        NodeState nodeState = NodeState.valueOf(StringUtils.substringBetween(components[0], "state[", "]"));
        boolean check = Boolean.parseBoolean(components[1]);

        return (state, i) -> state.get(i, nodeState) == check;
    }
}
