package consumer;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

import es.bsc.compss.api.COMPSs;
import model.Text;
import model.TextCollectionIndex;
import model.TextStats;
import storage.StorageItf;


public class Wordcount {

	private static TextStats[] partialResult;

	// To store parameters
	private static String textColAlias = null;
	private static String configPropertiesFile = null;
	private static int timesPerText = 1;
	// Default wcop = text makes stats to be persistent and returns them
	private static WCOP wcop = WCOP.WC_MAKE_PERS;
	// Default rtop = merged stats are returned
	private static RTOP rtop = RTOP.RT_IN;


	/**
	 * Checks the received application arguments
	 * 
	 * @param args
	 * @return
	 */
	private static boolean getAndCheckArguments(String[] args) {
		if (args.length < 1) {
			System.err.println("[ERROR] Bad arguments. " + getUsage());
			System.err.println("Available wordcount ops: " + Arrays.asList(WCOP.values()));
			System.err.println("Available reduce ops: " + Arrays.asList(RTOP.values()));
			return false;
		}

		// Parse arguments
		textColAlias = args[0];

		for (int argIndex = 1; argIndex < args.length;) {
			String arg = args[argIndex++];
			switch (arg) {
				case "-c":
					configPropertiesFile = args[argIndex++];
					File f = new File(configPropertiesFile);
					if (!f.exists() || f.isDirectory()) {
						System.err.println("[ERROR] Bad argument. Configuration file: " + configPropertiesFile + " does not exist.");
						return false;
					}
					break;
				case "-t":
					timesPerText = new Integer(args[argIndex++]);
					break;
				case "-wcop":
					wcop = WCOP.getOp(new Integer(args[argIndex++]));
					if (wcop == null) {
						System.err.println("[ERROR] Bad wordcount option " + wcop + ". Valid ones are: " + Arrays.asList(WCOP.values()));
						return false;
					}
					break;
				case "-rtop":
					rtop = RTOP.getOp(new Integer(args[argIndex++]));
					if (rtop == null) {
						System.err.println("[ERROR] Bad reducetask option " + rtop + ". Valid ones are: " + Arrays.asList(RTOP.values()));
						return false;
					}
					break;
				case "-h":
					System.err.println("[HELP] " + getUsage());
					break;
				default:
					System.err.println("[ERROR] Invalid argument");
					break;
			}
		}

		// All arguments retrieved
		return true;
	}

	/**
	 * Display usage
	 * 
	 * @return
	 */
	private static String getUsage() {
		StringBuilder sb = new StringBuilder("");

		sb.append("Usage").append("\n");

		sb.append("\t").append(Wordcount.class.getName()).append("\n");

		// Parameters
		sb.append("\t").append("\t").append("<text_col_alias>").append("\n"); // Alias for TextCol. Mandatory
		sb.append("\t").append("\t").append("[-t <times_per_text>]").append("\n"); // Times per text
		sb.append("\t").append("\t").append("[-c <config_properties>]").append("\n"); // Only if compss is not used
		sb.append("\t").append("\t").append("[-wcop <wc_id>]").append("\n"); // WCOP
		sb.append("\t").append("\t").append("[-rtop <rt_id>]").append("\n"); // RTOP
		sb.append("\t").append("\t").append("[-h]").append("\n"); // Display usage
		sb.append("\t").append("\t").append("\n");

		return sb.toString();
	}

	/**
	 * Log received arguments
	 * 
	 */
	private static void logArguments() {
		System.out.println("[LOG] Application Arguments:");
		System.out.println("[LOG]   textColAlias = " + textColAlias);
		System.out.println("[LOG]   configFile =   " + configPropertiesFile);
		System.out.println("[LOG]   timePerText =  " + timesPerText);
		System.out.println("[LOG]   WCOP =         " + wcop);
		System.out.println("[LOG]   RTOP =          " + rtop);
	}

	/**
	 * MAIN METHOD
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String args[]) throws Exception {
		System.out.println("[LOG] Retrieving arguments");
		
		// Get and check arguments
		boolean success = getAndCheckArguments(args);
		if (!success) {
			System.out.println("[LOG] Arguments failed");
			return;
		}

		// Log received parameters
		logArguments();

		// Load StorageItf (only for non-COMPSs executions)
		if (configPropertiesFile != null) {
			StorageItf.init(configPropertiesFile);
		}

		/* *****************************************************
		 * INITIALIZATION 
		 * *****************************************************/
		// Init texts to parse
		TextCollectionIndex tc = new TextCollectionIndex(0, textColAlias);
		System.out.println("[LOG] Obtained TextCollection: " + textColAlias);
		Text[] textsToCount = totalTexts(tc, timesPerText);
		partialResult = new TextStats[textsToCount.length];

		// This is not necessary if we use wordCount based on text.wordCountMakingPersistent
		if (wcop == WCOP.WC_FILL_IN || wcop == WCOP.WC_FILL_INOUT || wcop == WCOP.WC_IMPLICIT) {
			initAndStorePartialResults(textsToCount);
		}

		/* *****************************************************
		 * RUN WORDCOUNT 
		 * *****************************************************/
		System.out.println("[LOG] Computing result");

		long startTime = System.currentTimeMillis();
		TextStats finalResult = run(textsToCount);
		// Explicit synchronization
		COMPSs.waitForAllTasks();
		long endTime = System.currentTimeMillis();
		System.out.println("[TIMER] Elapsed time 1st: " + (endTime - startTime) + " ms");

		startTime = System.currentTimeMillis();
		finalResult = run(textsToCount);
		// Explicit synchronization
		COMPSs.waitForAllTasks();
		endTime = System.currentTimeMillis();
		System.out.println("[TIMER] Elapsed time 2nd: " + (endTime - startTime) + " ms");

		/* *****************************************************
		 * ENDERS 
		 * *****************************************************/
		System.out.println("[LOG] Final result contains " + finalResult.getSize() + " unique words.");
		System.out.println("[LOG] Result summary: ");
		System.out.println(finalResult.getSummary(10));

		System.out.println("[LOG] Clean results: ");
		startTime = System.currentTimeMillis();
		for (TextStats ts : partialResult) {
			try {
				ts.deletePersistent();
			} catch (Exception e) {
				System.err.println("Could not clean TextStats, oid = " + ts.getID() + " Exception: " + e.getMessage());
			}
		}
		endTime = System.currentTimeMillis();
		System.out.println("[TIMER] Clean time: " + (endTime - startTime) + " ms");

		// Stop StorageItf (only for non-COMPSs executions)
		if (configPropertiesFile != null) {
			StorageItf.finish();
		}
	}

	public static TextStats run(final Text[] texts) throws Exception {
		// MAP-WORDCOUNT
		System.out.println("[LOG] TEXTS LENGTH = " + texts.length);
		for (int i = 0; i < texts.length; i++) {
			switch(wcop) {
				case WC_FILL_IN:
					partialResult[i] = WordcountImpl.wordCountFillStatsIN(texts[i], partialResult[i]);
					break;
				case WC_FILL_INOUT:
					WordcountImpl.wordCountFillStatsINOUT(texts[i], partialResult[i]);
					break;
				case WC_MAKE_PERS:
					partialResult[i] = WordcountImpl.wordCountNewStats(texts[i]);
					break;
				case WC_IMPLICIT:
					partialResult[i].wordCountFillingStats(texts[i]);
					break;
				default:
					System.err.println("[ERROR] Bad Wordcount op " + wcop);
					return null;
			}

			// partialResult[i] = wordCountNewStats(texts[i]);
			// System.out.println("[LOG] After Wordcount, ID of stats (should not be null) " + partialResult[i].getID());
		}

		// REDUCE-MERGE
		System.out.println("[LOG] REDUCE");
		LinkedList<Integer> q = new LinkedList<Integer>();
		for (int i = 0; i < texts.length; i++) {
			q.add(i);
		}
		int x = 0;
		while (!q.isEmpty()) {
			x = q.poll();
			int y;
			if (!q.isEmpty()) {
				y = q.poll();
				switch(rtop) {
					case RT_IN:
						partialResult[x] = WordcountImpl.reduceTaskIN(partialResult[x], partialResult[y]);
						break;
					case RT_INOUT:
						WordcountImpl.reduceTaskINOUT(partialResult[x], partialResult[y]);
						break;
					case RT_IMPLICIT:
						partialResult[x].reduceTask(partialResult[y]);
						break;
					default:
						System.err.println("[ERROR] Bad ReduceTask op " + rtop);
						return null;
				}

				q.add(x);
			}
		}

		return partialResult[x];
	}

	/**
	 * Init texts to be counted
	 */
	private static Text[] totalTexts(TextCollectionIndex tc, int timesPerText) throws Exception {
		ArrayList<String> textTitles = tc.getTextTitles();
		int actualTexts = textTitles.size();
		int totalTexts = actualTexts * timesPerText;
		Text[] result = new Text[totalTexts];
		int index = 0;
		for (String textTitle : textTitles) {
			for (int j = 0; j < timesPerText; j++) {
				System.out.println("Initialization index = " + index);
				result[index] = new Text(0, textTitle);
				index++;
			}
		}
		return result;
	}

	private static void initAndStorePartialResults(Text[] textsToCount) {
		int i = 0;
		for (Text t : textsToCount) {
			partialResult[i] = new TextStats();
			System.out.println("[LOG] Creating empty partialResult in " + t.getLocation());
			partialResult[i].makePersistent(true, t.getLocation());
			System.out.println("[LOG] Empty partial result created with ID: " + partialResult[i].getID() + " in "
					+ partialResult[i].getLocation());
			i++;
		}

	}


	/**
	 * ENUMS for Operation Selection
	 *
	 */
	public static enum WCOP {
		WC_FILL_IN, 
		WC_FILL_INOUT, 
		WC_MAKE_PERS,
		WC_IMPLICIT;

		public static WCOP getOp(int id) {
			switch (id) {
				case 1:
					return WC_FILL_IN;
				case 2:
					return WC_FILL_INOUT;
				case 3:
					return WC_MAKE_PERS;
				case 4:
					return WC_IMPLICIT;
				default:
					return null;
			}
		}
	}

	public static enum RTOP {
		RT_IN, 
		RT_INOUT,
		RT_IMPLICIT;

		public static RTOP getOp(int id) {
			switch (id) {
				case 1:
					return RT_IN;
				case 2:
					return RT_INOUT;
				case 3:
					return RT_IMPLICIT;
				default:
					return null;
			}
		}
	}
	
}
