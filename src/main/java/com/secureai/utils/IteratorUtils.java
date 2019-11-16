package com.secureai.utils;

import java.util.Collection;
import java.util.Iterator;
import java.util.Set;

public class IteratorUtils {
    public static <T> T getInIterator(Iterator<T> iterator, int i) {
        for (int j = 0; iterator.hasNext(); j++) {
            if (i == j)
                return iterator.next();
            iterator.next();
        }
        return null;
    }

    public static <T> T getInCollection(Collection<T> collection, int i) {
        return IteratorUtils.getInIterator(collection.iterator(), i);
    }

    public static <T> T getInSet(Set<T> set, int i) {
        return IteratorUtils.getInIterator(set.iterator(), i);
    }
}
