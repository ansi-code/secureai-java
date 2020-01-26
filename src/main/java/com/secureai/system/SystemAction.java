package com.secureai.system;

import com.secureai.model.actionset.Action;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {

    private String resourceId;
    private String actionId;

    public void run(SystemEnvironment environment) {
        Action action = environment.getActionSet().getActions().get(this.actionId);
        if (action.getPreCondition().run(environment.getSystemState(), this.resourceId))
            action.getPostCondition().run(environment.getSystemState(), this.resourceId);
    }

}
