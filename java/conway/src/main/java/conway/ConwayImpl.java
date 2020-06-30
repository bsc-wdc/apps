package conway;

import conway.Zone;
import conway.Block;
//import es.bsc.compss.api.COMPSs;

public class ConwayImpl {
	public static void update_block (Zone Z, Block T) {
		for (int i = Conway.B_SIZE; i < 2*Conway.B_SIZE; ++i) {
			for (int j = Conway.B_SIZE; j < 2*Conway.B_SIZE; ++j) {
				int count = 0;
				
				for (int off_i = -1; off_i <= 1; ++off_i)
					for(int off_j = -1; off_j <= -1; ++off_j)
						if(Z.get(i+off_i,j+off_j) == 1)
							++count;
				
				if(Z.get(i,j) == 1) {
					if(count == 2 || count == 3)
						T.set(i-Conway.B_SIZE, j-Conway.B_SIZE, 1);
					else
						T.set(i-Conway.B_SIZE, j-Conway.B_SIZE, 0);
				} else {
					if(count == 3)
						T.set(i-Conway.B_SIZE, j-Conway.B_SIZE, 1);
					else
						T.set(i-Conway.B_SIZE, j-Conway.B_SIZE, 0);
				}
			}
		}
	}
}
