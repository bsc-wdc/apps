
package data.tree;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;


public class SplitNode extends Node implements Externalizable {

    int splitFeature;
    double splitValue;
    Tree lowerChild;
    Tree upperChild;


    public SplitNode() {

    }

    public SplitNode(int splitFeature, double splitValue, Tree lowerChild, Tree upperChild) {
        this.splitFeature = splitFeature;
        this.splitValue = splitValue;
        this.lowerChild = lowerChild;
        this.upperChild = upperChild;
    }

    @Override
    public void print(String firstPrefix, String nextPrefix) {
        if (lowerChild != null) {
            String value = ((double) ((int) (splitValue * 10_000)) / 10_000) + "";
            System.out.println(firstPrefix + splitFeature + "(" + value + ")");
            lowerChild.print(nextPrefix + "     ├── ", nextPrefix + "     │   ");
            upperChild.print(nextPrefix + "     └── ", nextPrefix + "         ");
        } else {
            System.out.println(firstPrefix);
        }

    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(splitFeature);
        oo.writeDouble(splitValue);
        oo.writeObject(lowerChild);
        oo.writeObject(upperChild);
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        splitFeature = oi.readInt();
        splitValue = oi.readDouble();
        lowerChild = (Tree) oi.readObject();
        upperChild = (Tree) oi.readObject();
    }

}
