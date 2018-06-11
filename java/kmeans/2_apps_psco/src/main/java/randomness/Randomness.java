package randomness;

import java.util.Random;

public class Randomness {
	private static final int RANDOMSEED = 0;
	private static final Random seedsProvider = new Random(RANDOMSEED);
	public static final int muSeed;
	
	static {
		muSeed = seedsProvider.nextInt();
	}
	
	public static int nextInt() {
		return seedsProvider.nextInt();
	}
}
