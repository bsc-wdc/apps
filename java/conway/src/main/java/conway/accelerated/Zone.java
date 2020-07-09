package conway.accelerated;
/*
import java.io.Serializable;

public class Zone implements Serializable {
	// Serializer version code
	private static final long serialVersionUID = 2L;

	private Block[][] supra;
	private int bSize;
	
	//Empty
	public Zone() {
		// Nothing to do. Only for serialization.
	}
	
	//With base
	//NOTE: To use only in a scope where Conway.WB and Conway.LB are properly defined
	public Zone(Block[][] ref, int iBlock, int jBlock) {
		this.supra = new Block[3][3];
		this.bSize = ref[iBlock][jBlock].getBSize();
		
		for (int off_i = 0; off_i < 3; ++off_i) {
			for (int off_j = 0; off_j < 3; ++off_j) {
				int i = (iBlock + off_i - 1 + Conway.WB) % Conway.WB;
				int j = (jBlock + off_j - 1 + Conway.LB) % Conway.LB;
				this.supra[off_i][off_j] = ref[i][j];
			}
		}
	}
	
	//Hard copy
    public Zone(Zone ref) {
    	this.supra = new Block[3][3];
    	this.bSize = ref.bSize;
    	
    	for(int i = 0; i < 3; ++i) {
    		for(int j = 0; j < 3; ++j) {
    			this.supra[i][j] = new Block(ref.supra[i][j]);
    		}
    	}
    }
    
    //Setters and Getters
	public int get(int i, int j) {
		return this.supra[i / this.bSize][j / this.bSize].get(i % this.bSize, j % this.bSize);
	}
	
    public void set(int i, int j, int val) {
        this.supra[i / this.bSize][j / this.bSize].set(i % this.bSize, j % this.bSize, val);
    }

	public int getBSize() {
		return this.bSize;
	}
	
	public Block getCenter() {
    	return this.supra[1][1];
    }
}
*/