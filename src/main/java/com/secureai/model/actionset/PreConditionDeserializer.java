package com.secureai.model.actionset;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.deser.std.StdDeserializer;
import com.secureai.model.stateset.State;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class PreConditionDeserializer extends StdDeserializer<Action.PreNodeStateFunction> {

    public PreConditionDeserializer() {
        this(null);
    }

    public PreConditionDeserializer(Class<?> vc) {
        super(vc);
    }

    @Override
    public Action.PreNodeStateFunction deserialize(JsonParser jsonParser, DeserializationContext deserializationContext) throws IOException {
        return this.parsePreConditions(jsonParser.getValueAsString());
    }

    private Action.PreNodeStateFunction parsePreConditions(String str) {
        System.out.println(str);
        if (str == null || str.equals("~") || str.equals("null"))
            return (state, i) -> true;

        List<Action.PreNodeStateFunction> andConditions = new ArrayList<>();
        if (!str.contains(" && "))
            andConditions.add(this.parsePreCondition(str));
        else {
            for (String andConditionString : str.split(" && ")) {
                List<Action.PreNodeStateFunction> orConditions = new ArrayList<>();
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

    private Action.PreNodeStateFunction parsePreCondition(String str) {
        String[] components = str.split(" == ");
        State nodeState = State.valueOf(StringUtils.substringBetween(components[0], "state[", "]"));
        boolean check = Boolean.parseBoolean(components[1]);

        return (state, i) -> state.get(i, nodeState) == check;
    }
}
