package producer;

import java.util.ArrayList;
import java.util.List;

// import dataclay.collections.DataClayArrayList;
import model.Text;
import model.TextCollection;
import model.TextCollectionIndex;
import storage.StorageItf;
import util.ids.StorageLocationID;


public class RemoteTextCollectionGenerator {

	private static int nodesPerLevel = 0;
	private static int chunkSize = 0;
	private static int timesPerFile = 1;

	public static boolean prepareForDebug = false;


	public static void main(String[] args) throws Exception {
		if (args.length < 3) {
			printErrorUsage();
			return;
		}

		if (!setOptionalArguments(args)) {
			return;
		}

		StorageItf.init(args[0]);
		final String textColAlias = args[1];
		final String remoteFilePath = args[2];

		TextCollectionIndex textCollection;
		try {
			textCollection = new TextCollectionIndex(0, textColAlias);
			System.out.println("[LOG] Found collection index with " + textCollection.getSize() + " files");
		} catch (Exception ex) {
			System.out.println("[LOG] No previous collection index found");
			ArrayList<TextCollection> tcs = new ArrayList<TextCollection>();
			int id = 1;
			for (StorageLocationID locID : StorageItf.getBackends().keySet()) {
				String prefixForTexts = textColAlias + id;
				TextCollection tc = new TextCollection(prefixForTexts, prepareForDebug);
				tc.makePersistent(prefixForTexts, locID);
				System.out.println("[LOG] Collection created at " + tc.getLocation());
				tcs.add(tc);
				id++;
			}
			textCollection = new TextCollectionIndex(tcs);
			textCollection.makePersistent(textColAlias);
			System.out.println("[LOG] Created new collection index");
		}
		System.out.println("[LOG] Collection index located at " + textCollection.getLocation());

		// DataClayArrayList<String> emptyList = new
		// DataClayArrayList<String>(nodesPerLevel, chunkSize);
		ArrayList<String> emptyList = new ArrayList<String>();
		List<String> textTitles = new ArrayList<String>();
		for (int i = 0; i < timesPerFile; i++) {
			textTitles = textCollection.addTextsFromPath(remoteFilePath, emptyList);

			System.out.println("[LOG] Updated collection. Now it has " + textCollection.getSize() + " files.");
			for (String textTitle : textTitles) {
				Text t = new Text(0, textTitle);
				System.out.println("[LOG] New text " + textTitle + " is located at " + t.getLocation());
			}
		}
		StorageItf.finish();

	}

	private static boolean setOptionalArguments(String[] args) {
		for (int argIndex = 3; argIndex < args.length;) {
			String arg = args[argIndex++];
			if (arg.equals("-n")) {
				nodesPerLevel = new Integer(args[argIndex++]);
				if (nodesPerLevel <= 0) {
					System.err.println("Bad argument. NodesPerLevel must be greater than zero");
					return false;
				}
			} else if (arg.equals("-c")) {
				chunkSize = new Integer(args[argIndex++]);
				if (chunkSize <= 0) {
					System.err.println("Bad argument. ChunkSize must be greater than zero");
					return false;
				}
			} else if (arg.equals("-t")) {
				timesPerFile = new Integer(args[argIndex++]);
				if (timesPerFile <= 0) {
					System.err.println("Bad argument. TimesPerFile must be greater than zero");
					return false;
				}
			} else if (arg.equals("-debug")) {
				prepareForDebug = true;
			} else {
				printErrorUsage();
				return false;
			}
		}
		if ((chunkSize > 0 && nodesPerLevel == 0) || (nodesPerLevel > 0 && chunkSize == 0)) {
			System.err.println("Error. Missing argument. If NodesPerLevel is set, ChunkSize must be set too");
			return false;
		}

		return true;
	}

	private static void printErrorUsage() {
		System.err.println("Bad arguments. Usage: \n\n" + RemoteTextCollectionGenerator.class.getName()
				+ " <config_properties> <text_col_alias> <remote_path> [-t <times_file>] [-debug] "
				+ "[-n <nodes_per_level> -c <chunk_size>] (-n and -c are only for dataClay list tests) \n");
	}

}