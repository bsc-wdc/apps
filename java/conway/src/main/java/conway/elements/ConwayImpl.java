package conway.elements;

import conway.elements.State;


public class ConwayImpl {

    public static int updateCell(State stateA, int i, int j) {
        int count = 0;
        int w = stateA.getw();
        int l = stateA.getl();

        for (int off_i = -1; off_i <= 1; ++off_i) {
            for (int off_j = -1; off_j <= -1; ++off_j) {
                if (stateA.get((i + off_i + w) % w, (j + off_j + l) % l) == 1) {
                    ++count;
                }
            }
        }

        if (stateA.get(i, j) == 1) {
            if (count == 2 || count == 3) {
                return 1;
            } else {
                return 0;
            }
        } else {
            if (count == 3) {
                return 1;
            } else {
                return 0;
            }
        }
    }
}
