package model;

import java.util.HashMap;
import java.util.Map.Entry;

import serialization.DataClayObject;


@SuppressWarnings("serial")
public class TextStats extends DataClayObject {

	private HashMap<String, Integer> currentWordCount;


	public TextStats() {
		currentWordCount = new HashMap<String, Integer>();
		// System.out.println("[ TextStats ] Call to empty constructor (for COMPSs, to fill results later)");
	}

	public TextStats(HashMap<String, Integer> newWordCount) {
		currentWordCount = new HashMap<String, Integer>();
		currentWordCount.putAll(newWordCount);
		// System.out.println("[ TextStats ] Call to constructor for wordcount");
	}

	public void setCurrentWordCount(HashMap<String, Integer> newWordCount) {
		currentWordCount.putAll(newWordCount);
	}

	public HashMap<String, Integer> getCurrentWordCount() {
		return currentWordCount;
	}

	public int getSize() {
		return currentWordCount.size();
	}

	public void mergeWordCounts(final TextStats newWordCount) {
		HashMap<String, Integer> wordCountToMerge = newWordCount.getCurrentWordCount();
		for (Entry<String, Integer> entry : wordCountToMerge.entrySet()) {
			String word = entry.getKey();
			Integer count = entry.getValue();
			Integer curCount = currentWordCount.get(word);
			if (curCount == null) {
				currentWordCount.put(word, count);
			} else {
				currentWordCount.put(word, curCount + count);
			}
		}
	}

	public HashMap<String, Integer> getSummary(int maxEntries) {
		int i = 0;
		HashMap<String, Integer> result = new HashMap<String, Integer>();
		for (Entry<String, Integer> curEntry : currentWordCount.entrySet()) {
			result.put(curEntry.getKey(), curEntry.getValue());
			i++;
			if (i == maxEntries) {
				break;
			}
		}
		return result;
	}

	// Task
	public void wordCountFillingStats(Text text) {
		text.wordCountFillingStats(this);
	}
	
	// Task
	public void reduceTask(TextStats m2) {
		this.mergeWordCounts(m2);
	}

}
