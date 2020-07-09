package conway.accelerated;

import conway.accelerated.Block;

public class ConwayImpl {

	public static Block updateBlock(Block b00, Block b01, Block b02, Block b10, Block b11, Block b12, Block b20,
			Block b21, Block b22, int aFactor, int bSize) {

		// Rebuild zone

		Block[][] subStateA = new Block[3][3];
		Block[][] subStateB = new Block[3][3];
		Block[][] subStateC;
		
		//subStateA
		subStateA = new Block[3][3];
		for (int off_i = 0; off_i < 3; ++off_i) {
			for (int off_j = 0; off_j < 3; ++off_j) {
				subStateA[off_i][off_j] = new Block(bSize);
			}
		}

		// subStateB
		subStateB[0][0] = new Block(b00);
		subStateB[0][1] = new Block(b01);
		subStateB[0][2] = new Block(b02);
		subStateB[1][0] = new Block(b10);
		subStateB[1][1] = new Block(b11);
		subStateB[1][2] = new Block(b12);
		subStateB[2][0] = new Block(b20);
		subStateB[2][1] = new Block(b21);
		subStateB[2][2] = new Block(b22);
		
		//	iterations 
		for (int t = aFactor; t >= 0; --t) {
			subStateC = subStateA;
			subStateA = subStateB;
			subStateB = subStateC;

			for (int i = bSize - t; i < 2 * bSize + t; ++i) {
				for (int j = bSize - t; j < 2 * bSize + t; ++j) {

					int count = 0;

					// Count
					for (int off_i = -1; off_i <= 1; ++off_i) {
						for (int off_j = -1; off_j <= 1; ++off_j) {
							if (off_i != 0 || off_j != 0) {
								Block p = subStateA[(i + off_i) / bSize][(j + off_j) / bSize];
								if (p.get((i + off_i) % bSize, (j + off_j) % bSize) == 1) {
									++count;
								}
							}
						}
					}

					// Rules
					Block p = subStateA[i / bSize][j / bSize];
					Block q = subStateB[i / bSize][j / bSize];
					int mod_i = i % bSize;
					int mod_j = j % bSize;

					if (p.get(mod_i, mod_j) == 1) {
						if (count == 2 || count == 3) {
							q.set(mod_i, mod_j, 1);
						} else {
							q.set(mod_i, mod_j, 0);
						}
					} else {
						if (count == 3) {
							q.set(mod_i, mod_j, 1);
						} else {
							q.set(mod_i, mod_j, 0);
						}
					}

				}
			}
		}
		
		return subStateB[1][1];
	}

}
