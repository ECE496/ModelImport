package com.example.modelimport;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import org.tensorflow.lite.Interpreter;

import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    Interpreter tflite;
    Bitmap bmp;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        try {
            tflite = new Interpreter(loadModelFile());
            InputStream is = this.getAssets().open("face.jpg");
            bmp = BitmapFactory.decodeStream(is);
        } catch (Exception e) {
            e.printStackTrace();
        }

        doInference(bmp);



    }

    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void doInference(Bitmap bmp){
        float [][][][] img = new float[1][200][200][3];
        float [][] output = new float[1][7];
        for (int i = 0; i < 200; i++){
            for (int j = 0; j < 200; j++){
                int p = bmp.getPixel(j, i);
                img[0][i][j][0] = ((p >> 16) & 0xff) / (float)255;
                img[0][i][j][1] = ((p >> 8) & 0xff) / (float)255;
                img[0][i][j][2] = (p & 0xff) / (float)255;
            }
        }
        tflite.run(img, output);
    }
}
