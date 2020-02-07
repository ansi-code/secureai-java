package com.secureai.rl.qn;

import com.secureai.utils.ArrayUtils;
import com.secureai.utils.RandomUtils;
import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

public class FilteredDynamicQTable extends DynamicQTable {

    @Getter
    @Setter
    private DynamicQTableGetFilter dynamicQTableGetFilter;

    public FilteredDynamicQTable(int actionSpaceSize) {
        super(actionSpaceSize);
    }

    @Override
    public double[] get(int state) {
        double[] result = this.dynamicQTableGetFilter != null ? ArrayUtils.multiply(super.get(state), this.dynamicQTableGetFilter.run(state)) : super.get(state);
        if (Arrays.stream(result).max().orElse(Double.NEGATIVE_INFINITY) == Double.NEGATIVE_INFINITY)
            result[RandomUtils.getRandom(0, result.length - 1)] = .5;
        return result;
    }

    public interface DynamicQTableGetFilter {
        double[] run(int state);
    }
}
