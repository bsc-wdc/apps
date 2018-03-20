package model;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import serialization.DataClayObject;


// import util.ids.StorageLocationID;

@SuppressWarnings("serial")
public class TextCollection extends DataClayObject {

	private String textPrefix;
	private ArrayList<String> textTitles;
	private boolean debug;


	public TextCollection(String prefixForTextsInCollection, boolean doDebug) {
		textTitles = new ArrayList<String>();
		textPrefix = prefixForTextsInCollection;
		debug = doDebug;
	}

	public String getTextPrefix() {
		return textPrefix;
	}

	public ArrayList<String> getTextTitles() {
		return textTitles;
	}

	public int getSize() {
		return textTitles.size();
	}

	public String addTextFromFile(final String filePath, List<String> emptyList) throws IOException {
		String textTitle = textPrefix + ".file" + (textTitles.size() + 1);
		Text t = new Text(textTitle, emptyList, debug);
		t.makePersistent(textTitle);
		textTitles.add(textTitle);
		t.addWords(filePath);
		return textTitle;
	}

	/**
	 * public String addTextFromFileToLocation(final String filePath, List <String> emptyList, final StorageLocationID
	 * location) throws IOException { String textTitle = textPrefix + ".file" + (textTitles.size() + 1); Text t = new
	 * Text(textTitle, emptyList); t.makePersistent(textTitle, location); textTitles.add(textTitle);
	 * t.addWords(filePath); return textTitle; }
	 */
}
