package krunal.com.example.cameraapp;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import com.google.android.material.floatingactionbutton.FloatingActionButton;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final int REQUEST_STORAGE_PERMISSION = 1;

    private static final String FILE_PROVIDER_AUTHORITY = "com.example.android.fileprovider";

    private AppExecutor mAppExcutor;

    private ImageView mImageView;

    private Button mStartCamera;

    private String mTempPhotoPath;

    private Bitmap mResultsBitmap;

    private FloatingActionButton mClear,mSave,mShare;

    private ObjectDetector objectDetector; // Add this line

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mAppExcutor = new AppExecutor();

        mImageView = findViewById(R.id.imageView);
        mClear = findViewById(R.id.clear);
        mSave = findViewById(R.id.Save);
        mShare = findViewById(R.id.Share);
        mStartCamera = findViewById(R.id.startCamera);

        mImageView.setVisibility(View.GONE);
        mShare.setVisibility(View.GONE);
        mSave.setVisibility(View.GONE);
        mClear.setVisibility(View.GONE);

        objectDetector = new ObjectDetector(this); // Add this line

        mStartCamera.setOnClickListener(v -> {
            // Check for the external storage permission
            if (ContextCompat.checkSelfPermission(getApplicationContext(),
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    != PackageManager.PERMISSION_GRANTED) {

                // If you do not have permission, request it
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
                        REQUEST_STORAGE_PERMISSION);
            } else {
                // Launch the camera if the permission exists
                launchCamera();
            }
        });

        mSave.setOnClickListener((View v) -> {
            mAppExcutor.diskIO().execute(() -> {
                // Delete the temporary image file
                BitmapUtils.deleteImageFile(this, mTempPhotoPath);

                // Save the image
                BitmapUtils.saveImage(this, mResultsBitmap);

            });

            Toast.makeText(this,"Image Save",Toast.LENGTH_LONG).show();

        });

        mClear.setOnClickListener(v -> {
            // Clear the image and toggle the view visibility
            mImageView.setImageResource(0);
            mStartCamera.setVisibility(View.VISIBLE);
            mSave.setVisibility(View.GONE);
            mShare.setVisibility(View.GONE);
            mClear.setVisibility(View.GONE);

            mAppExcutor.diskIO().execute(() -> {
                // Delete the temporary image file
                BitmapUtils.deleteImageFile(this, mTempPhotoPath);
            });

        });

        mShare.setOnClickListener((View v) -> {

            mAppExcutor.diskIO().execute(() ->{
                // Delete the temporary image file
                BitmapUtils.deleteImageFile(this, mTempPhotoPath);

                // Save the image
                BitmapUtils.saveImage(this, mResultsBitmap);

            });

            // Share the image
            BitmapUtils.shareImage(this, mTempPhotoPath);

        });

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        // Called when you request permission to read and write to external storage
        switch (requestCode) {
            case REQUEST_STORAGE_PERMISSION: {
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // If you get permission, launch the camera
                    launchCamera();
                } else {
                    // If you do not get permission, show a Toast
                    Toast.makeText(this, R.string.permission_denied, Toast.LENGTH_SHORT).show();
                }
                break;
            }
        }
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // If the image capture activity was called and was successful
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // Process the image and set it to the TextView
            processAndSetImage();
        } else {
            // Otherwise, delete the temporary image file
            BitmapUtils.deleteImageFile(this, mTempPhotoPath);
        }
    }


    /**
     * Creates a temporary image file and captures a picture to store in it.
     */
    private void launchCamera() {

        // Create the capture image intent
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the temporary File where the photo should go
            File photoFile = null;
            try {
                photoFile = BitmapUtils.createTempImageFile(this);
            } catch (IOException ex) {
                // Error occurred while creating the File
                ex.printStackTrace();
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {

                // Get the path of the temporary file
                mTempPhotoPath = photoFile.getAbsolutePath();

                // Get the content URI for the image file
                Uri photoURI = FileProvider.getUriForFile(this,
                        FILE_PROVIDER_AUTHORITY,
                        photoFile);

                // Add the URI so the camera can store the image
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);

                // Launch the camera activity
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
            }
        }
    }

    /**
     * Method for processing the captured image and setting it to the TextView.
     */
    private void processAndSetImage() {

        // Toggle Visibility of the views
        mStartCamera.setVisibility(View.GONE);
        mSave.setVisibility(View.VISIBLE);
        mShare.setVisibility(View.VISIBLE);
        mClear.setVisibility(View.VISIBLE);
        mImageView.setVisibility(View.VISIBLE);

        // Resample the saved image to fit the ImageView
        mResultsBitmap = BitmapUtils.resamplePic(this, mTempPhotoPath);

        // Perform object detection
        List<ObjectDetector.Recognition> recognitions = objectDetector.recognizeImage(mResultsBitmap);

        // Draw bounding boxes on the bitmap
        mResultsBitmap = drawBoundingBoxes(mResultsBitmap, recognitions);

        // Set the new bitmap to the ImageView
        mImageView.setImageBitmap(mResultsBitmap);
    }

    private Bitmap drawBoundingBoxes(Bitmap bitmap, List<ObjectDetector.Recognition> recognitions) {
        Bitmap mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(mutableBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(8.0f);

        Paint textPaint = new Paint();
        textPaint.setColor(Color.RED);
        textPaint.setTextSize(36.0f);

        for (ObjectDetector.Recognition recognition : recognitions) {
            RectF location = recognition.getLocation();
            canvas.drawRect(location, paint);
            canvas.drawText(recognition.getLabel() + ": " + String.format("%.2f", recognition.getConfidence()), location.left, location.top, textPaint);
        }

        return mutableBitmap;
    }
}

