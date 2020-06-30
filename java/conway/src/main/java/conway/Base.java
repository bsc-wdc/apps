package conway;

import java.io.Serializable;
import conway.Block;

@SuppressWarnings("serial")
public class Base implements Serializable {
	private Block supra[][];
	
	//Random
	Base () {
		supra = new Block[Conway.WB][Conway.LB];
		for (int I = 0; I < Conway.WB; ++I)
			for (int J = 0; J < Conway.LB; ++J)
				supra[I][J] = new Block();
	}
	
	//Hard copy
	Base (Base ref) {
		supra = new Block[Conway.WB][Conway.LB];
		for (int I = 0; I < Conway.WB; ++I)
			for (int J = 0; J < Conway.LB; ++J)
				this.supra[I][J] = new Block(ref.supra[I][J]);
	}
	
	//Setters 
	public void set (int I, int J, Block T) {
		supra[I][J] = T;
	}
	
	public void set_hard (int I, int J, Block T) {
		supra[I][J] = new Block(T);
	}
	
	//Getters
	public Block get (int I, int J) {
		return supra[I][J];
	}
	
	//Print
	public void print () {
		for (int I = 0; I < Conway.WB; ++I) {
			for (int i = 0; i < Conway.B_SIZE; ++i) {
				for (int J = 0; J < Conway.LB; ++J)
					for (int j = 0; j < Conway.B_SIZE; ++j)
						System.out.print(supra[I][J].get(i, j));
				System.out.println();
			}
		}
		System.out.println();
	}
	
	//Swap
	public static void swap(Base A, Base B) {
		Block C;
		for (int I = 0; I < Conway.WB; ++I) {
			for (int J = 0; J < Conway.LB; ++J) {
				C = A.supra[I][J];
				A.supra[I][J] = B.supra[I][J];
				B.supra[I][J] = C;
			}
		}
	}
}
