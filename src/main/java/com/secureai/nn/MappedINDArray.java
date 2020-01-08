package com.secureai.nn;

import com.secureai.utils.Decorated;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Map;

public class MappedINDArray extends Decorated<INDArray> {
    private Map<String, Integer> map;
    private int blockSize = 1; // Default is block size of 1

    public MappedINDArray(INDArray value, Map<String, Integer> map) {
        super(value);
        this.map = map;
        this.blockSize = (int) (value.length() / map.size());
    }

    public INDArray get(String key) {
        return this.get().get(this.intervalOf(key));
    }

    public void set(String key, INDArray values) {
        this.get().put(new INDArrayIndex[]{this.intervalOf(key)}, values);
    }

    public INDArrayIndex intervalOf(String key) {
        int index = this.map.get(key);
        return NDArrayIndex.interval(index * this.blockSize, (index + 1) * this.blockSize);
    }
}
