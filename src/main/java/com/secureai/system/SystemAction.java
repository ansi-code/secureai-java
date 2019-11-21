package com.secureai.system;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SystemAction {

    private Integer nodeIndex;
    private NodeAction nodeAction;

    public void run(SystemState systemState) {
        if (this.nodeAction.getDefinition().getPreNodeStateFunction().run(systemState, nodeIndex))
            this.nodeAction.getDefinition().getPostNodeStateFunction().run(systemState, nodeIndex);
    }

}
