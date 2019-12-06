package com.secureai.utils;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class Nd4jUtils {
    public static INDArray hInsert(INDArray a, INDArray b, int index) {
        if (index == 0)
            return Nd4j.hstack(b, a.get(NDArrayIndex.all(), NDArrayIndex.interval(index, 1, a.columns())));
        else if (index == a.columns())
            return Nd4j.hstack(a.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 1, index)), b);

        return Nd4j.hstack(a.get(NDArrayIndex.all(), NDArrayIndex.interval(0, 1, index)), b, a.get(NDArrayIndex.all(), NDArrayIndex.interval(index, 1, a.columns())));
    }
}
