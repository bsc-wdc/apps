package model.chunked;

import java.util.ArrayList;

import serialization.DataClayObject;


@SuppressWarnings("serial")
public class Chunk extends DataClayObject {

    private ArrayList<float[]> points;


    public Chunk() {
        points = new ArrayList<float[]>();
    }

    public void addVector(float[] vector) {
        float[] newVector = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            newVector[i] = vector[i];
        }
        points.add(newVector);
    }

    public float[] getVector(int pos) {
        return points.get(pos);
    }

    public int getSize() {
        return points.size();
    }

    public int getDimensionsPerVector() {
        return points.get(0).length;
    }

    public void updateVector(int pos, float[] vector) {
        float[] newVector = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            newVector[i] = vector[i];
        }
        points.set(pos, vector);
    }
    
}
