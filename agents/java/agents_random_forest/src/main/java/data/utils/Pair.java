package data.utils;

public class Pair {

    private int classId;
    private int count;


    public Pair() {
        // Nothing to do
    }

    public Pair(int classId, int count) {
        this.classId = classId;
        this.count = count;
    }

    public int getClassId() {
        return this.classId;
    }

    public int getCount() {
        return this.count;
    }

    public void setClassId(int classId) {
        this.classId = classId;
    }

    public void setCount(int count) {
        this.count = count;
    }

    @Override
    public String toString() {
        return "<" + classId + "->" + count + ">";
    }
}
