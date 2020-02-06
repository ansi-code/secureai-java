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
        double[] result = ArrayUtils.multiply(super.get(state), this.dynamicQTableGetFilter.run(state));
        if (Arrays.stream(result).max().orElse(0d) == 0d)
            result[RandomUtils.getRandom(0, result.length - 1)] = .5;
        return result;
    }

    public interface DynamicQTableGetFilter {
        double[] run(int state);
    }
}
