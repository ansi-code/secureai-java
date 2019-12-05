package com.secureai.rl.qn;

import lombok.Getter;
import lombok.Setter;

public class FilteredDynamicQTable extends DynamicQTable {

    @Getter
    @Setter
    private DynamicQTableGetFilter dynamicQTableGetFilter;

    public FilteredDynamicQTable(int actionSpaceSize) {
        super(actionSpaceSize);
    }

    @Override
    public double[] get(int state) {
        return this.dynamicQTableGetFilter.run(state, super.get(state));
    }

    public interface DynamicQTableGetFilter {
        double[] run(int state, double[] output);
    }
}
