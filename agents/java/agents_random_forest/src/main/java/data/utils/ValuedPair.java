package data.utils;

public class ValuedPair {

    private int id;
    private Double value;
    private int classId;


    public ValuedPair() {
        // Nothing to do
    }

    public ValuedPair(int id, double value, int classId) {
        this.id = id;
        this.value = value;
        this.classId = classId;
    }

    public int getId() {
        return this.id;
    }

    public Double getValue() {
        return this.value;
    }

    public int getClassId() {
        return this.classId;
    }

    public void setId(int id) {
        this.id = id;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public void setClassId(int classId) {
        this.classId = classId;
    }

    @Override
    public String toString() {
        return this.id + "(" + this.value + " -> " + this.classId + ")";
    }
}
