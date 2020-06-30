package conway;

import conway.State;
//import es.bsc.compss.api.COMPSs;

public class ConwayImpl {
	
	
	public static int update_cell (State state_A, int i, int j) {
		int count = 0;
		int w = state_A.getw();
		int l = state_A.getl();
		
		for (int off_i = -1; off_i <= 1; ++off_i)
			for(int off_j = -1; off_j <= -1; ++off_j)
				if(state_A.get((i+off_i+w)%w,(j+off_j+l)%l) == 1)
					++count;
		
		if(state_A.get(i,j) == 1) {
			if(count == 2 || count == 3)
				return 1;
			else
				return 0;
		} else {
			if(count == 3)
				return 1; 
			else
				return 0;
		}
	}
}
