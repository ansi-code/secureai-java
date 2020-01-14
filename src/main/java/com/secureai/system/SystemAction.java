package com.secureai.system;

import com.secureai.model.actionset.Action;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {

    private String resourceId;
    private Action action;

    public void run(SystemState systemState) {
        if (this.action.getPreCondition().run(systemState, resourceId))
            this.action.getPostCondition().run(systemState, resourceId);
    }

}
