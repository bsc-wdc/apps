package consumer;

import model.Text;
import model.TextStats;


public class WordcountImpl {

	/**
	 * wordcount that returns an internally created (and made persistent) stats object
	 * 
	 * @param text
	 * @return
	 */
	public static TextStats wordCountNewStats(Text text) {
		return text.wordCountMakingPersistent();
	}

	/**
	 * wordcount filling already created stats object and returns it
	 * 
	 * @param text
	 * @param result
	 * @return
	 */
	public static TextStats wordCountFillStatsIN(Text text, TextStats result) {
		text.wordCountFillingStats(result);
		return result;
	}

	/**
	 * wordcount filling already created stats but does not return it
	 * 
	 * @param text
	 * @param result
	 */
	public static void wordCountFillStatsINOUT(Text text, TextStats result) {
		text.wordCountFillingStats(result);
	}

	/**
	 * reducetask assuming the result is updated in the first parameter and returns it
	 * 
	 * @param m1
	 * @param m2
	 * @return
	 */
	public static TextStats reduceTaskIN(TextStats m1, TextStats m2) {
		m1.mergeWordCounts(m2);
		return m1;
	}

	/**
	 * reducetask assuming the result is updated in the first parameter but does not return it
	 * 
	 * @param m1
	 * @param m2
	 */
	public static void reduceTaskINOUT(TextStats m1, TextStats m2) {
		m1.mergeWordCounts(m2);
	}

}
