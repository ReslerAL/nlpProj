package code;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class LSTMModel {
	
	//add tensorflow binaries
	static {
		System.load("/home/alon/workspace/tensorflowTesl/lib/jni/libtensorflow_jni.so");
	}
	
	/*
	 * how to use example
	 */
	public static void main(String[] args) throws NumberFormatException, IOException {
		LSTMModel model = new LSTMModel("/home/alon/workspace/nlp/saved/1503569471412/pbmodel", 300, "/home/alon/workspace/nlp/vocabulary.txt");
		String q = "who was the next minister of transport after c.d. howe dklsfmsdlk";
		float[] res = model.apply(q);
		for(int i = 0; i < res.length; i++) {
			System.out.print(String.format("%f ", res[i]));
		}
	}
	
	private SavedModelBundle b;
	private Session sess;
	private int embedSize;
	private Map<String, Integer> vocab;

	/**
	 * load the model from the model directory and the vocabulary from the vocabulary file
	 * @throws IOException 
	 * @throws NumberFormatException 
	 */
	public LSTMModel(String dir, int embedSize, String vocabFilePath) throws NumberFormatException, IOException {
		b = SavedModelBundle.load(dir, "serve");
		sess = b.session();
		this.embedSize = embedSize;
		this.vocab = loadVocabulary(vocabFilePath);
	}
	
	
	/**
	 * 
	 * @param sentence - list of indices which is the sentence according to vocabulary of embedding
	 * @return - the sentence embedding according to the model
	 */
	public float[] apply(int[] sentence) {
		if (sentence.length == 0) {
			return null;
		}
		int[][] data = {sentence}; //the model expect for batches
		int[] sent_len = {sentence.length};
		Tensor x = Tensor.create(data);
		Tensor x_seq_len = Tensor.create(sent_len);
		float[] result = sess.runner()
	            .feed("x", x).feed("x_seq_len", x_seq_len)
	            .fetch("embd")
	            .run()
	            .get(0)
	            .copyTo(new float[1][embedSize])[0];
		return result;
	}
	
	/**
	 * same as above function just here we first convert the sentence from string to embedding list according to vocab
	 * @param sentence - input sentence as string
	 * @return - the sentence embedding according to the model
	 */
	public float[] apply(String sentence) {
		int[] sent = sentToIndx(sentence);
		return apply(sent);
	}
	

	private int[] sentToIndx(String sentence) {
		
		String[] words = sentence.split(" ");
		int[] result = new int[words.length];
		for (int i = 0; i < words.length; i++) {
			Integer idx = this.vocab.get(words[i]);
			if (idx == null) {
				result[i] = this.vocab.size();
			}
			else {
				result[i] = idx;
			}
		}
		return result;
	}


	private Map<String, Integer> loadVocabulary(String filePath) throws NumberFormatException, IOException {
		Map<String, Integer> map = new HashMap<>();
		String fileName = "/home/alon/workspace/nlp/vocabulary.txt";
		try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
		    String line;
		    while ((line = br.readLine()) != null) {
		    	String[] entry = line.split(" ");
		    	map.put(entry[0], Integer.parseInt(entry[1]));
		    }
		}
		return map;
	}


}
