package com.xck.tensor5;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private TextView tvRes;
    private DrawView drawView;
    private DrawModel drawModel;
    private Button btnClear;
    private Button btnDetect;

    private static final int PIXEL_WIDTH = 28;
    private BaseClassifer classifier;
    private Executor executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //init
        btnClear = findViewById(R.id.btnClear);
        btnDetect = findViewById(R.id.btnDetect);
        tvRes = findViewById(R.id.tv_res);
        btnClear.setOnClickListener(this);
        btnDetect.setOnClickListener(this);
        btnDetect.setVisibility(View.INVISIBLE);

        drawModel = new DrawModel(PIXEL_WIDTH, PIXEL_WIDTH);
        drawView = findViewById(R.id.view_draw);
        drawView.setModel(drawModel);

        initTensorflowAndLoadModel();
    }

    @Override
    protected void onResume() {
        drawView.onResume();
        super.onResume();
    }

    @Override
    protected void onPause() {
        drawView.onPause();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }


    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.btnClear:
                drawModel.clear();
                drawView.reset();
                drawView.invalidate();
                tvRes.setText("");
                break;
            case R.id.btnDetect:
                float[] pixels = drawView.getPixelData();
                final List<BaseClassifer.Recognition> results = classifier.recognizeImage(pixels);
                if(results.size() > 0){
                    String value = "识别到的数字是：" + results.get(0).getTitle();
                    tvRes.setText(value);
                }
                break;
        }
    }

    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String MODEL_FILE = "file:///android_asset/mnist_model_graph.pb";
    private static final String LABEL_FILE = "file:///android_asset/mnist_labels.txt";

    private void initTensorflowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = ImageClassifer.create(
                            getAssets(),
                            MODEL_FILE,
                            LABEL_FILE,
                            INPUT_SIZE,
                            INPUT_NAME,
                            OUTPUT_NAME);
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            btnDetect.setVisibility(View.VISIBLE);
                        }
                    });
                } catch (final IOException e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }
}
