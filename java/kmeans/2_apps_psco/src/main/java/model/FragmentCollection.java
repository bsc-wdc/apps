package model;

import java.util.ArrayList;

import serialization.DataClayObject;


@SuppressWarnings("serial")
public class FragmentCollection extends DataClayObject {

    private ArrayList<Fragment> fragments;


    public FragmentCollection() {
        fragments = new ArrayList<Fragment>();
    }

    public FragmentCollection(int foo, final String alias) {
        super(alias);
    }

    public void addFragment(Fragment f) {
        fragments.add(f);
    }

    public int getNumFragments() {
        return this.fragments.size();
    }

    public int getVectorsPerFragment() {
        return this.fragments.get(0).getNumVectors();
    }

    public int getNumDimensionsPerVector() {
        return this.fragments.get(0).getDimensionsPerVector();
    }

    public ArrayList<Fragment> getFragments() {
        return fragments;
    }

    public int getTotalVectors() {
        int counter = 0;
        for (Fragment f : fragments) {
            counter += f.getNumVectors();
        }
        return counter;
    }
    
}
