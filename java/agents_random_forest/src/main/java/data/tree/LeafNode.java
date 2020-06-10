package data.tree;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.LinkedList;


public class LeafNode extends Node implements Externalizable {

    private LinkedList<Pair> frequency;


    public LeafNode() {
        this.frequency = new LinkedList<>();
    }

    public LeafNode(int[] probability) {
        this.frequency = new LinkedList<>();
        for (int i = 0; i < probability.length; i++) {
            if (probability[i] != 0) {
                this.frequency.add(new Pair(i, probability[i]));
            }
        }
    }

    @Override
    public void print(String firstPrefix, String nextPrefix) {
        System.out.println(firstPrefix + this.frequency);
    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(this.frequency.size());
        for (Pair p : this.frequency) {
            oo.writeInt(p.classId);
            oo.writeInt(p.count);
        }
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        int count = oi.readInt();
        for (int i = 0; i < count; i++) {
            Pair p = new Pair();
            p.classId = oi.readInt();
            p.count = oi.readInt();
            this.frequency.add(p);
        }
    }


    private class Pair {

        private int classId;
        private int count;


        public Pair() {
        }

        public Pair(int classId, int count) {
            this.classId = classId;
            this.count = count;
        }

        @Override
        public String toString() {
            return "<" + classId + "->" + count + ">";
        }
    }
}
