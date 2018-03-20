package consumer;

import java.util.ArrayList;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;

import model.Text;
import model.TextCollectionIndex;
import model.TextStats;
import storage.CallbackEvent;
import storage.CallbackHandler;
import storage.StorageItf;


public class WordcountAsync {

	public static Text[] textsToCount;
	public static String[] textsLocations;
	public static TextStats finalResult;
	public static int timesPerText = 1;


	public static void main(String args[]) throws Exception {
		if (!checkArguments(args)) {
			return;
		}

		StorageItf.init(args[0]);
		String textColAlias = args[1];
		if (args.length == 3) {
			timesPerText = new Integer(args[2]);
		}

		// Init texts to parse
		TextCollectionIndex tc = new TextCollectionIndex(0, textColAlias);
		System.out.println("[LOG] Obtained TextCollection: " + textColAlias);
		initVariables(tc, timesPerText);

		// Run wordcount
		System.out.println("[LOG] Computing result");
		long startTime = System.currentTimeMillis();
		run();
		long endTime = System.currentTimeMillis();
		System.out.println("[TIMER] Elapsed time 1st: " + (endTime - startTime) + " ms");
		startTime = System.currentTimeMillis();
		run();
		endTime = System.currentTimeMillis();
		System.out.println("[TIMER] Elapsed time 2nd: " + (endTime - startTime) + " ms");

		System.out.println("[LOG] Final result contains " + finalResult.getSize() + " unique words.");
		System.out.println("[LOG] Result summary: ");
		System.out.println(finalResult.getSummary(10));

		long totalWCTime = 0;
		for (int i = 0; i < wcTimes.length; i++) {
			System.out.println("[LOG] wcTimes[" + i + "] == " + wcTimes[i]);
			totalWCTime += wcTimes[i];
		}
		System.out.println("[LOG] Total WCTime = " + totalWCTime);
		long totalRTTime = 0;
		for (int i = 0; i < reduceTimes.length; i++) {
			System.out.println("[LOG] reduceTimes[" + i + "] == " + reduceTimes[i]);
			totalRTTime += reduceTimes[i];
		}
		System.out.println("[LOG] Total WCTime = " + totalRTTime);

		StorageItf.finish();
	}


	static CountDownLatch c1;
	static CountDownLatch c2;
	static AtomicInteger textsDone;
	static long[] wcTimes;
	static long[] reduceTimes;


	public static void run() throws Exception {
		wcTimes = new long[textsToCount.length];
		reduceTimes = new long[textsToCount.length];
		textsDone = new AtomicInteger(0);
		c1 = new CountDownLatch(textsToCount.length);
		c2 = new CountDownLatch(textsToCount.length - 1);
		final ConcurrentLinkedQueue<TextStats> results = new ConcurrentLinkedQueue<TextStats>();
		final long initTime = System.currentTimeMillis();
		for (int i = 0; i < textsToCount.length; i++) {
			StorageItf.executeTask(textsToCount[i].getID(), "wordCountMakingPersistent()Lmodel/TextStats;", new Object[] {},
					textsLocations[i], new CallbackHandler() {

						@Override
						protected void eventListener(CallbackEvent arg0) {
							int curText = textsDone.incrementAndGet();
							wcTimes[curText - 1] = System.currentTimeMillis() - initTime;
							TextStats result = (TextStats) arg0.getContent();
							// System.out.println("[LOG] Arrived results " +
							// curText + " with id " + result.getID());
							if (curText == textsToCount.length) { // last
								try {
									c2.await();
								} catch (InterruptedException e) {
									e.printStackTrace();
								}
								TextStats inqueue = results.poll();
								while (inqueue != null) { // merge with pending
									long start = System.currentTimeMillis();
									result.mergeWordCounts(inqueue);
									reduceTimes[curText - 1] += (System.currentTimeMillis() - start);
									inqueue = results.poll();
								}
								finalResult = result; // set result
								// System.out.println("[LOG] Offering " +
								// finalResult.getID() + " size: " +
								// finalResult.getSize());
							} else {
								TextStats inqueue = results.poll();
								while (inqueue != null) {
									// System.out.println("[LOG] found " +
									// inqueue.getID() + " in queue");
									long start = System.currentTimeMillis();
									result.mergeWordCounts(inqueue);
									reduceTimes[curText - 1] += (System.currentTimeMillis() - start);
									inqueue = results.poll();
								}
								// System.out.println("[LOG] appending " +
								// result.getID() + " to queue");
								results.offer(result);
								c2.countDown();
							}
							c1.countDown();
						}
					});
			// System.out.println("[LOG] Sent request " + i);
		}
		c1.await();
	}

	/**
	 * Init texts to be counted
	 */
	private static void initVariables(TextCollectionIndex tc, int timesPerText) throws Exception {
		ArrayList<String> textTitles = tc.getTextTitles();
		int actualTexts = textTitles.size();
		int totalTexts = actualTexts * timesPerText;
		textsToCount = new Text[totalTexts];
		textsLocations = new String[totalTexts];
		int index = 0;
		for (String textTitle : textTitles) {
			for (int j = 0; j < timesPerText; j++) {
				textsToCount[index] = new Text(0, textTitle); // get by alias
				textsLocations[index] = StorageItf.getLocation(textsToCount[index].getID());
				index++;
			}
		}
	}

	private static boolean checkArguments(String[] args) {
		if (args.length < 2) {
			System.err.println("[ERROR] Bad arguments. " + getUsage());
			return false;
		}

		for (int argIndex = 1; argIndex < args.length;) {
			String arg = args[argIndex++];
			if (arg.equals("-t")) {
				timesPerText = new Integer(args[argIndex++]);
			} else if (arg.equals("-h")) {
				System.out.println("[HELP] " + getUsage());
				return false;
			}
		}

		return true;
	}

	private static String getUsage() {
		return "Usage \n\n" + WordcountAsync.class.getName() + " <config_properties> <text_col_alias> [-t <times_per_text>] [-h] \n";
	}
}
