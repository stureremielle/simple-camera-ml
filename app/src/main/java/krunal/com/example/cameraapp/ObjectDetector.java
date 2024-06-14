package krunal.com.example.cameraapp;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class ObjectDetector {
    private Interpreter tfliteInterpreter;
    private List<String> labels;
    private Context context;

    public ObjectDetector(Context context) {
        this.context = context;
        loadModel();
    }

    private void loadModel() {
        try {
            AssetFileDescriptor fileDescriptor = context.getAssets().openFd("model.tflite");
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            Interpreter.Options options = new Interpreter.Options();
            tfliteInterpreter = new Interpreter(modelBuffer, options);
            labels = loadLabels();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private List<String> loadLabels() throws IOException {
        List<String> labelList = new ArrayList<>();
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("labelmap.txt");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        int length = inputStream.available();
        byte[] buffer = new byte[length];
        inputStream.read(buffer);
        inputStream.close();
        String[] labelArray = new String(buffer).split("\n");
        for (String label : labelArray) {
            labelList.add(label.trim());
        }
        return labelList;
    }

    public List<Recognition> recognizeImage(Bitmap bitmap) {
        int inputSize = 300; // Adjust according to your model input size
        int[] intValues = new int[inputSize * inputSize];
        float[][][] outputLocations = new float[1][10][4]; // Adjust based on your model output shape
        float[][] outputClasses = new float[1][10]; // Adjust based on your model output shape
        float[][] outputScores = new float[1][10]; // Adjust based on your model output shape
        float[] outputNumDetections = new float[1]; // Adjust based on your model output shape

        // Pre-process the image into input tensor
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true);
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());
        tfliteInterpreter.run(intValues, new Object[]{outputLocations, outputClasses, outputScores, outputNumDetections});

        // Process outputs and return recognitions
        List<Recognition> recognitions = new ArrayList<>();
        for (int i = 0; i < outputNumDetections[0]; i++) {
            RectF detection = new RectF(
                    outputLocations[0][i][1] * inputSize,
                    outputLocations[0][i][0] * inputSize,
                    outputLocations[0][i][3] * inputSize,
                    outputLocations[0][i][2] * inputSize);
            int labelOffset = (int) outputClasses[0][i];
            String label = labels.get(labelOffset + 1); // Index 0 is background
            float confidence = outputScores[0][i];

            Recognition recognition = new Recognition(detection, label, confidence);
            recognitions.add(recognition);
        }
        return recognitions;
    }

    public static class Recognition {
        private RectF location;
        private String label;
        private float confidence;

        public Recognition(RectF location, String label, float confidence) {
            this.location = location;
            this.label = label;
            this.confidence = confidence;
        }

        public RectF getLocation() {
            return location;
        }

        public String getLabel() {
            return label;
        }

        public float getConfidence() {
            return confidence;
        }
    }
}

