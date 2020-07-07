package conway.blocks;

import conway.blocks.Block;
import conway.blocks.Zone;


public class ConwayImpl {

    public static Block updateBlock(Zone z) {
    	int bSize = z.getBSize();
        Block res = new Block(bSize);

        for (int i = bSize; i < 2 * bSize; ++i) {
            for (int j = bSize; j < 2 * bSize; ++j) {
            	
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
                        res.set(i - bSize, j - bSize, 1);
                    } else {
                        res.set(i - bSize, j - bSize, 0);
                    }
                } else {
                    if (count == 3) {
                        res.set(i - bSize, j - bSize, 1);
                    } else {
                        res.set(i - bSize, j - bSize, 0);
                    }
                }
            }
        }
        
        return res;
    }

}
