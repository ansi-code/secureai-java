package com.secureai.rl.qn;

import com.secureai.utils.ArrayUtils;
import com.secureai.utils.NumberUtils;
import com.secureai.utils.RandomUtils;
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
        double[] result = this.dynamicQTableGetFilter != null ? ArrayUtils.replaceNaN(ArrayUtils.multiply(super.get(state), this.dynamicQTableGetFilter.run(state)), Double.NEGATIVE_INFINITY) : super.get(state);
        if (!NumberUtils.hasValue(ArrayUtils.max(result)))
            result[RandomUtils.getRandom(0, result.length - 1)] = 0;
        return result;
    }

    public interface DynamicQTableGetFilter {
        double[] run(int state);
    }
}
