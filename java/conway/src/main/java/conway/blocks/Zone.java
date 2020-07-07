package conway.blocks;

import java.io.Serializable;

public class Zone implements Serializable {

	// Serializer version code
	private static final long serialVersionUID = 2L;

	private Block[][] supra;
	private int bSize;

	public Zone() {
		// Nothing to do. Only for serialization.
	}

	public Zone(Block[][] ref, int iBlock, int jBlock) {
		this.supra = new Block[3][3];
		this.bSize = ref[0][0].getBSize();

		for (int off_i = 0; off_i < 3; ++off_i)
			for (int off_j = 0; off_j < 3; ++off_j) {
				int i = (iBlock + off_i - 1 + Conway.WB) % Conway.WB;
				int j = (jBlock + off_j - 1 + Conway.LB) % Conway.LB;
				this.supra[off_i][off_j] = ref[i][j];
			}
	}

	public int get(int i, int j) {
		return this.supra[i / this.bSize][j / this.bSize].get(i % this.bSize, j % this.bSize);
	}

	public int getBSize() {
		return bSize;
	}
}
