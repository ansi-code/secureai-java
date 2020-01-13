package com.secureai.system;

import com.secureai.model.actionset.Action;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {

    private Integer nodeIndex;
    private Action action;

    public void run(SystemState systemState) {
        if (this.action.getPreCondition().run(systemState, nodeIndex))
            this.action.getPostCondition().run(systemState, nodeIndex);
    }

}
