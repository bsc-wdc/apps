package data.utils;

import java.util.Comparator;


public class ComparatorValuedPair implements Comparator<ValuedPair> {

    @Override
    public int compare(ValuedPair t, ValuedPair t1) {
        int d = Double.compare(t.getValue(), t1.getValue());
        if (d == 0) {
            d = -1;
        }
        return d;
    }
}