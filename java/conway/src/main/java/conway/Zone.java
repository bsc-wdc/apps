package conway;

import java.io.Serializable;

@SuppressWarnings("serial")
public class Zone implements Serializable {
	private Block supra[][];
	
	Zone () {
		
	}
	
	Zone (Base ref, int I, int J) {
		supra = new Block[3][3];
				
		for (int off_i = 0; off_i < 3; ++off_i)
			for (int off_j = 0; off_j < 3; ++off_j) {
				int i = (I+off_i-1+Conway.WB)%Conway.WB;
				int j = (J+off_j-1+Conway.LB)%Conway.LB;
				this.supra[off_i][off_j] = ref.get(i,j);
			}
	}

	public int get(int i, int j) {
		return supra[i/Conway.B_SIZE][j/Conway.B_SIZE].get(i%Conway.B_SIZE, j%Conway.B_SIZE);
	}
}
