package com.secureai.rl.rl4j;

import lombok.Value;
import org.deeplearning4j.rl4j.space.Encodable;

@Value
public class SystemState implements Encodable {

    int i;
    int step;

    @Override
    public double[] toArray() {
        double[] ar = new double[1];
        ar[0] = (20 - i);
        return ar;
    }

}
