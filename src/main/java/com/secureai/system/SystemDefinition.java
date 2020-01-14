package com.secureai.system;

import com.secureai.model.topology.Topology;
import com.secureai.utils.BidirectionalMap;
import com.secureai.utils.IteratorUtils;
import com.secureai.utils.StreamUtils;
import lombok.Getter;

import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SystemDefinition {
    @Getter
    private Topology topology;
    @Getter
    private BidirectionalMap<String, Integer> resourcesMap;

    public SystemDefinition(Topology topology) {
        this.topology = topology;
        this.topology.getResources().entrySet().stream().flatMap(entry -> IntStream.range(0, entry.getValue().getReplication()).mapToObj(i -> String.format("%s[%d]", entry.getKey(), i))).forEach(System.out::println);

        this.resourcesMap = StreamUtils.fromIterator(IteratorUtils.zipWithIndex(this.topology.getResources().keySet().iterator())).collect(Collectors.toMap(
                Map.Entry::getValue,
                Map.Entry::getKey,
                (u, v) -> {
                    throw new IllegalStateException(String.format("Duplicate key %s", u));
                },
                BidirectionalMap::new));
    }

    public long getInConnectionsCount(String resourceId) {
        return this.topology.getConnections().values().stream().filter(edge -> edge.getTo().equals(resourceId)).count();
    }

    public long getOutConnectionsCount(String resourceId) {
        return this.topology.getConnections().values().stream().filter(edge -> edge.getFrom().equals(resourceId)).count();
    }

    public void prettyPrint() {
        System.out.print("\033[H\033[2J");
        System.out.flush();
        int j = 0;
        for (String nodeName : this.topology.getResources().keySet()) {
            for (int i = 0; i < this.topology.getResources().get(nodeName).getReplication(); i++) {
                System.out.print(nodeName);
                System.out.print(String.format(" (%d)  ", j));
            }
            System.out.print("  ");
            j++;
        }
        System.out.print("\n");
    }
}
