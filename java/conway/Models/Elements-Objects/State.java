package conway;

import java.io.Serializable;


@SuppressWarnings("serial")
public class State implements Serializable {
	//Matrix
	private int w, l;
	private int[][] matrix;
	
	//Getters and setters
	public void set (int i, int j, int val) {
		this.matrix[i][j] = val;
	}
	
	public int get (int i, int j) {
		return this.matrix[i][j];
	}
	
	public int getw() {
		return this.w;
	}
	
	public int getl() {
		return this.l;
	}
	
	//Void
	public State () {
		
	}
	
	//Random
	public State(int w, int l) {
		this.w = w;   this.l = l;
		this.matrix = new int[w][l];
		
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j< l; ++j)
				this.matrix[i][j] = (int) Math.floor(Math.random()*2);
		}
	}
	
	//Hard copy
	public State(State ref) {
		this.w = ref.w;   this.l = ref.l;
		this.matrix = new int[w][l];
		
		for (int i = 0; i < w; ++i) {
			for (int j = 0; j < l; ++j)
				this.matrix[i][j] = ref.matrix[i][j];
		}
	}
	
	//Print
	public void print() {
		for (int i = 0; i < this.w; ++i) {
			for (int j = 0; j< this.l; ++j)
				System.out.print(this.matrix[i][j]);
			System.out.println();
		}
		System.out.println();
	}
}