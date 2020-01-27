package com.secureai.rl.qn;

import com.secureai.utils.ArrayUtils;
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
        return ArrayUtils.multiply(super.get(state), this.dynamicQTableGetFilter.run(state));
    }

    public interface DynamicQTableGetFilter {
        double[] run(int state);
    }
}
