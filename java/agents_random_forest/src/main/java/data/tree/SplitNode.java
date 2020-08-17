package data.tree;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;


public class SplitNode extends Node implements Externalizable {

    private int splitFeature;
    private double splitValue;
    private Tree lowerChild;
    private Tree upperChild;


    public SplitNode() {
        // Only for externalisation
    }

    public SplitNode(int splitFeature, double splitValue, Tree lowerChild, Tree upperChild) {
        this.splitFeature = splitFeature;
        this.splitValue = splitValue;
        this.lowerChild = lowerChild;
        this.upperChild = upperChild;
    }

    @Override
    public void print(String firstPrefix, String nextPrefix) {
        if (this.lowerChild != null) {
            String value = ((double) ((int) (this.splitValue * 10_000)) / 10_000) + "";
            System.out.println(firstPrefix + this.splitFeature + "(" + value + ")");
            this.lowerChild.print(nextPrefix + "     ├── ", nextPrefix + "     │   ");
            this.upperChild.print(nextPrefix + "     └── ", nextPrefix + "         ");
        } else {
            System.out.println(firstPrefix);
        }

    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(this.splitFeature);
        oo.writeDouble(this.splitValue);
        oo.writeObject(this.lowerChild);
        oo.writeObject(this.upperChild);
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        this.splitFeature = oi.readInt();
        this.splitValue = oi.readDouble();
        this.lowerChild = (Tree) oi.readObject();
        this.upperChild = (Tree) oi.readObject();
    }

}
