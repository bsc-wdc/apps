package conway.accelerated;

import java.io.Serializable;

public class Block implements Serializable {
	// Serializer version code
	private static final long serialVersionUID = 2L;

	// Matrix
	private int[][] matrix;
	private int bSize;

	// Empty
	public Block() {
		// Nothing to do. Only for serialization.
	}

	// Random
	public Block(int bSize) {
		this.bSize = bSize;
		this.matrix = new int[this.bSize][this.bSize];

		for (int i = 0; i < this.bSize; ++i) {
			for (int j = 0; j < this.bSize; ++j) {
				this.matrix[i][j] = (int) Math.floor(Math.random() * 2);
			}
		}
	}

	// Hard copy
	public Block(Block ref) {
		this.bSize = ref.bSize;
		this.matrix = new int[this.bSize][this.bSize];

		for (int i = 0; i < this.bSize; ++i) {
			for (int j = 0; j < this.bSize; ++j) {
				this.matrix[i][j] = ref.matrix[i][j];
			}
		}
	}

	// Getters and setters
	public void set(int i, int j, int val) {
		this.matrix[i][j] = val;
	}

	public int get(int i, int j) {
		return this.matrix[i][j];
	}

	public int getBSize() {
		return this.bSize;
	}
}