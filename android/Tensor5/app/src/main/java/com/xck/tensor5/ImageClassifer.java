package com.xck.tensor5;

import android.content.res.AssetManager;
import android.support.v4.os.TraceCompat;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class ImageClassifer implements BaseClassifer {
    private static final String TAG = "ImageClassifer";

    //Only return this many results with at least this confidence
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;  //阀

    //Config values;
    private String inputName;
    private String outputName;
    private int inputSize;

    //Pre-allocated buffers   预先分配的缓冲区
    private Vector<String> labels = new Vector<String>();
    private float[] outputs;
    private String[] outputNames;

    //tensorflow的接口
    private TensorFlowInferenceInterface tensorFlowInferenceInterface;

    private boolean runStats = false;

    private ImageClassifer(){
    }

    /**
     *
     * @param assetManager            The asset manager to be used to load assets.
     * @param modelFilename           The filepath of the model GraphDef protocol buffer.
     * @param labelFilename           The filepath of label file for classes.
     * @param inputSize               The input size. A square image of inputSize x inputSize is assumed.
     * @param inputName               The label of the image input node.
     * @param outputName              The label of the output node.
     * @throws java.io.IOException
     */
    public static BaseClassifer create(AssetManager assetManager,
            String modelFilename,
            String labelFilename,
            int inputSize,
            String inputName,
            String outputName) throws IOException {
        ImageClassifer c = new ImageClassifer();
        c.inputName = inputName;
        c.outputName = outputName;

        //Read the label names to memory
        String actualFilename = labelFilename.split("file:///android_asset/")[1];
        Log.i(TAG, "Reading label from :" + actualFilename);
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
        String line;
        while((line = br.readLine()) != null){
            c.labels.add(line);
        }
        br.close();

        c.tensorFlowInferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        //The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        int numClasses = (int)c.tensorFlowInferenceInterface.graph().operation(outputName).output(0).shape().size(1);
        Log.i(TAG, "Read" + c.labels.size() + "labels, output layer size is " + numClasses);
        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        //理想情况下，可以从输入操作的形状中检索inputSize。但是，通常使用的graphdef中用于输入的占位符节点不指定形状，
        // 因此它必须作为参数传入
        c.inputSize = inputSize;

        //Pre-allocate buffers.
        c.outputNames = new String[] {outputName};
        c.outputs = new float[numClasses];

        return c;
    }


    @Override
    public List<Recognition> recognizeImage(float[] pixels) {
        //Log this method so that it can be analyzed with systrace.
        TraceCompat.beginSection("recognizeImage");

        //copy the input data into Tensorflow
        TraceCompat.beginSection("feed");
        tensorFlowInferenceInterface.feed(inputName, pixels, new long[]{inputSize * inputSize});
        TraceCompat.endSection();;

        //Run the inference call.
        TraceCompat.beginSection("run");
        tensorFlowInferenceInterface.run(outputNames, runStats);
        TraceCompat.endSection();

        //Copy the output Tensor back into the output array.
        TraceCompat.beginSection("fetch");
        tensorFlowInferenceInterface.fetch(outputName, outputs);
        TraceCompat.endSection();

        //Find the best classifications.
        PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(3,
                new Comparator<Recognition>() {
            @Override
            public int compare(Recognition o1, Recognition o2) {
                //Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(o2.getConfidence(), o1.getConfidence());
            }
        });
        for(int i=0; i<outputs.length; i++){
            if(outputs[i] > THRESHOLD) {
                pq.add(new Recognition(""+i, labels.size()>1 ? labels.get(i) : "unknown", outputs[i], null));
            }
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for(int i=0; i<recognitionsSize; i++){
            recognitions.add(pq.poll());  //先get，再remove
        }
        TraceCompat.endSection();//"recognizeImage"
        return recognitions;
    }

    @Override
    public void enableStatLogging(boolean debug) {runStats = debug;}

    @Override
    public String getStatString() {
        return tensorFlowInferenceInterface.getStatString();
    }

    @Override
    public void close() {
        tensorFlowInferenceInterface.close();
    }
}
