package conway;

import java.io.Serializable;


@SuppressWarnings("serial")
public class Block implements Serializable {
	//Matrix
	private int[][] matrix;
	
	//Getters and setters
	public void set (int i, int j, int val) {
		this.matrix[i][j] = val;
	}
	
	public int get (int i, int j) {
		return this.matrix[i][j];
	}
	
	//Random
	public Block() {
		this.matrix = new int[Conway.B_SIZE][Conway.B_SIZE];
		
		for (int i = 0; i < Conway.B_SIZE; ++i) {
			for (int j = 0; j < Conway.B_SIZE; ++j)
				this.matrix[i][j] = (int) Math.floor(Math.random()*2);
		}
	}
	
	//Hard copy
	public Block(Block ref) {
		this.matrix = new int[Conway.B_SIZE][Conway.B_SIZE];
		
		for (int i = 0; i < Conway.B_SIZE; ++i) {
			for (int j = 0; j < Conway.B_SIZE; ++j)
				this.matrix[i][j] = ref.matrix[i][j];
		}
	}
}