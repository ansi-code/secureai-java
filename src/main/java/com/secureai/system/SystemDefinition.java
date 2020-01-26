package com.secureai.system;

import com.secureai.model.topology.Task;
import com.secureai.model.topology.Topology;
import lombok.Getter;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SystemDefinition {
    @Getter
    private Topology topology;

    @Getter
    private List<String> resources; // List of "taskId.replicationId"

    public SystemDefinition(Topology topology) {
        this.topology = topology;

        this.resources = this.topology.getTasks().entrySet().stream().flatMap(entry -> IntStream.range(0, entry.getValue().getReplication()).mapToObj(i -> String.format("%s.%d", entry.getKey(), i))).collect(Collectors.toList());
    }

    public long getInConnectionsCount(String resourceId) {
        String taskId = resourceId.substring(0, resourceId.lastIndexOf('.'));
        return this.topology.getConnections().values().stream().filter(edge -> edge.getTo().equals(taskId)).count();
    }

    public long getOutConnectionsCount(String resourceId) {
        String taskId = resourceId.substring(0, resourceId.lastIndexOf('.'));
        return this.topology.getConnections().values().stream().filter(edge -> edge.getFrom().equals(taskId)).count();
    }

    public Task getTask(String resourceId) {
        String taskId = resourceId.substring(0, resourceId.lastIndexOf('.'));
        return this.topology.getTasks().get(taskId);
    }

    public void prettyPrint() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
        int j = 0;
        for (String nodeName : this.topology.getTasks().keySet()) {
            for (int i = 0; i < this.topology.getTasks().get(nodeName).getReplication(); i++) {
                System.out.print(nodeName);
                System.out.print(String.format(" (%d)  ", j));
            }
            System.out.print("  ");
            j++;
        }
        System.out.print("\n");
    }
}
