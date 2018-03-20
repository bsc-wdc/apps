package producer;

import model.Text;
import model.TextCollectionIndex;
import storage.StorageItf;


public class CleanCurrentTextCollection {

	public static void main(String[] args) throws Exception {
		if (args.length != 2) {
			System.err.println("Bad arguments. Usage: \n\n" + CleanCurrentTextCollection.class.getName()
					+ " <config_properties> <text_col_alias> \n");
			return;
		}

		// Aarguments
		StorageItf.init(args[0]);
		final String textColAlias = args[1];

		try {
			TextCollectionIndex tc = new TextCollectionIndex(0, textColAlias);
			System.out.println("[LOG] Previous text collection found, deleting ...");
			for (String title : tc.getTextTitles()) {
				Text t = new Text(0, title);
				System.out.println("[LOG] Deleting found text " + t.getID() + " at " + t.getLocation());
				t.deletePersistent();
			}
			tc.deletePersistent();
			System.out.println("[LOG] Collection deleted.");
		} catch (Exception ex) {
			System.out.println("[LOG] No previous text collection found");
		}
		StorageItf.finish();
	}
}
