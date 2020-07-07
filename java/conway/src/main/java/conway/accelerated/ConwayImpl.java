package conway.accelerated;

import conway.accelerated.Block;
import conway.accelerated.Zone;

public class ConwayImpl {

	public static void updateBlock(Zone z, Block res, int aFactor) {

		Zone stateA = new Zone(z);
		Zone stateB = new Zone(z);
		Zone c;

		for (int t = aFactor; t >= 0; --t) {
			c = stateA;
			stateA = stateB;
			stateB = c;
			for (int i = z.getBSize() - t; i < 2 * z.getBSize() + t; ++i) {
				for (int j = z.getBSize() - t; j < 2 * z.getBSize() + t; ++j) {

					int count = 0;

					for (int off_i = -1; off_i <= 1; ++off_i) {
						for (int off_j = -1; off_j <= -1; ++off_j) {
							if (stateA.get(i + off_i, j + off_j) == 1) {
								++count;
							}
						}
					}

					if (stateA.get(i, j) == 1) {
						if (count == 2 || count == 3) {
							stateB.set(i, j, 1);
						} else {
							stateB.set(i, j, 0);
						}
					} else {
						if (count == 3) {
							stateB.set(i, j, 1);
						} else {
							stateB.set(i, j, 0);
						}
					}

				}
			}
		}
		
		res = new Block(stateB.getCenter());
	}

}
