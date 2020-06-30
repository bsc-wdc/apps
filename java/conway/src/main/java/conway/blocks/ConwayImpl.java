package conway.blocks;

import conway.blocks.Block;
import conway.blocks.Zone;


public class ConwayImpl {

    public static Block updateBlock(Zone z) {
        Block res = new Block();

        for (int i = Conway.B_SIZE; i < 2 * Conway.B_SIZE; ++i) {
            for (int j = Conway.B_SIZE; j < 2 * Conway.B_SIZE; ++j) {
                int count = 0;

                for (int off_i = -1; off_i <= 1; ++off_i) {
                    for (int off_j = -1; off_j <= -1; ++off_j) {
                        if (z.get(i + off_i, j + off_j) == 1) {
                            ++count;
                        }
                    }
                }

                if (z.get(i, j) == 1) {
                    if (count == 2 || count == 3) {
                        res.set(i - Conway.B_SIZE, j - Conway.B_SIZE, 1);
                    } else {
                        res.set(i - Conway.B_SIZE, j - Conway.B_SIZE, 0);
                    }
                } else {
                    if (count == 3) {
                        res.set(i - Conway.B_SIZE, j - Conway.B_SIZE, 1);
                    } else {
                        res.set(i - Conway.B_SIZE, j - Conway.B_SIZE, 0);
                    }
                }
            }
        }

        return res;
    }

}
