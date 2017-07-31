package cn.alexchao.multicnn;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.renderscript.RenderScript;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Scanner;

import messagepack.ParamUnpacker;
import network.CNNdroid;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class CifarActivity extends AppCompatActivity implements View.OnClickListener {
    // global references of UI components
    private Button mBtn;
    private TextView mText;

    // global references of CNNdroid
    private CNNdroid mConv;
    private RenderScript mRenderScript;
    private String[] mLabels;

    // global settings
    private final int mTextSize = 20;
    private final String mModelPath = "/storage/emulated/0/multicnn/netfiles/cifar/";
    private final String mBufferPath = "/storage/emulated/0/tmpImages/";

    private File imageFile;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cifar);

        initialize();

        new CifarActivity.prepareModel().execute(mRenderScript);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == Config.REQUEST_TAKE_PHOTO_CODE && resultCode == Activity.RESULT_OK
                && data != null) {
            performCrop(Uri.fromFile(imageFile));
//            Toast.makeText(this, "image taken", Toast.LENGTH_SHORT).show();
//            performCrop(data.getData());

//            Toast.makeText(this, "image saved", Toast.LENGTH_SHORT).show();
//            performCrop(Uri.fromFile(imageFile));
        } else if (requestCode == Config.REQUEST_CHOP_PHOTO_CODE && resultCode == Activity.RESULT_OK
                && data != null) {
            Toast.makeText(this, "image saved", Toast.LENGTH_SHORT).show();
            performInference(Uri.fromFile(imageFile));
            //performInference(data);
        }
    }

    // onClickListener
    @Override
    public void onClick(View v) {
        if (v.getId() == R.id.btn) {
            if (initFile()) {
                takePhoto();
            } else {
                Toast.makeText(this, "Initialization Failed", Toast.LENGTH_SHORT).show();
            }
        }
    }

    //  -- async task: read the model from files --
    private class prepareModel extends AsyncTask<RenderScript, Void, CNNdroid> {
        ProgressDialog progDialog;
        RelativeLayout layout = (RelativeLayout) findViewById(R.id.layout) ;

        protected void onPreExecute () {
            mText.setText("Loading Model Network Parameters...");
            mText.setTextSize(mTextSize);
            mBtn.setVisibility(View.GONE);
            layout.setClickable(false);
            layout.setFocusable(false);
            layout.setFocusableInTouchMode(false);
            progDialog = new ProgressDialog(CifarActivity.this);
            progDialog.setMessage("Please Wait...");
            progDialog.setIndeterminate(false);
            progDialog.setProgressStyle(ProgressDialog.STYLE_SPINNER);
            progDialog.setCancelable(true);
            progDialog.show();
        }

        @Override
        protected CNNdroid doInBackground(RenderScript... params) {
            long loadTime = System.currentTimeMillis();
            try {
                mConv = new CNNdroid(mRenderScript, mModelPath + "Cifar10_def.txt");
            } catch (Exception e) {
                e.printStackTrace();
            }
            loadTime = System.currentTimeMillis() - loadTime;
            Log.d("Async", "Time consume: " + loadTime);
            return mConv;
        }

        protected void onPostExecute(CNNdroid result) {
            mText.setText("\n\n\n\nPress \"Run\" to\nStart the Benchmark...");
            mText.setTextSize(mTextSize + 5);
            mText.setTextColor(Color.rgb(0, 0 ,0));
            mBtn.setText("Run");
            mBtn.setVisibility(View.VISIBLE);
            progDialog.dismiss();
            //layout.setBackground(getResources().getDrawable(R.drawable.back));
            layout.setClickable(true);
            layout.setFocusable(true);
            layout.setFocusableInTouchMode(true);
        }
    }

    // init operations
    private void initialize() {
        askForPermission();

        this.mBtn = (Button) findViewById(R.id.btn);
        this.mText = (TextView) findViewById(R.id.textView);
        mBtn.setOnClickListener(this);

        mRenderScript = RenderScript.create(this);

        readLabels();
    }

    private String accuracy(float[] input_matrix, String[] labels, int topk) {
        String result = "";
        int[] max_num = {-1, -1, -1, -1, -1};
        float[] max = new float[topk];
        for (int k = 0; k < topk ; ++k) {
            for (int i = 0; i < 10; ++i) {
                if (input_matrix[i] > max[k]) {
                    boolean newVal = true;
                    for (int j = 0; j < topk; ++j)
                        if (i == max_num[j])
                            newVal = false;
                    if (newVal) {
                        max[k] = input_matrix[i];
                        max_num[k] = i;
                    }
                }
            }
        }

        for (int i = 0 ; i < topk ; i++)
            result += labels[max_num[i]]  + " , P = " + max[i] * 100 + " %\n\n";
        return result;
    }

    private void performInference(Uri uri) {
        float[][][][] inputBatch = new float[1][3][32][32];
        Bitmap bmp;

        try {
            InputStream stream = getContentResolver().openInputStream(uri);
            bmp = BitmapFactory.decodeStream(stream);
            bmp = Bitmap.createScaledBitmap(bmp, 32, 32, false);
//            bmp = (Bitmap) data.getExtras().get("data");
//            //Bitmap bmp1 = Bitmap.createScaledBitmap(bmp, imgSize, imgSize, true);
//            Bitmap bmp2 = Bitmap.createScaledBitmap(bmp, 32, 32, false);
           // img.setImageBitmap(bmp1);
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Infer Error", Toast.LENGTH_SHORT).show();
            return;
        }

        ParamUnpacker pu = new ParamUnpacker();
        float[][][] mean = (float[][][]) pu.unpackerFunction(mModelPath + "mean.msg", float[][][].class);

        for (int j = 0; j < 32; ++j)
            for (int k = 0; k < 32; ++k) {
                int color = bmp.getPixel(j, k);
                inputBatch[0][0][k][j] = (float) (blue(color)) - mean[0][j][k];
                inputBatch[0][1][k][j] = (float) (green(color)) - mean[1][j][k];
                inputBatch[0][2][k][j] = (float) (red(color)) - mean[2][j][k];
            }
        float[][] output = (float[][]) mConv.compute(inputBatch);
        String resultStr = "\n" + accuracy(output[0], mLabels, 3);
        mText.setText(resultStr);
        mText.setTextSize(mTextSize);
    }

    //  -- operations on photos --
    private void takePhoto() {
        // setup system camera
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // save the photo into file
        intent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(imageFile));
        intent.putExtra("outputFormat", Bitmap.CompressFormat.PNG.toString());
        // start activity
        startActivityForResult(intent, Config.REQUEST_TAKE_PHOTO_CODE);
    }

    private void performCrop(Uri uri) {
        // chop the photo then replace the original photo with it
        Intent intent = new Intent("com.android.camera.action.CROP");
        intent.setDataAndType(uri, "image/*");
        intent.putExtra("crop", true);
        intent.putExtra("aspectX", 1);
        intent.putExtra("aspectY", 1);
        intent.putExtra("outputX", 128);
        intent.putExtra("outputY", 128);
        intent.putExtra("scale", false);
        intent.putExtra("return-data", true);
        intent.putExtra(MediaStore.EXTRA_OUTPUT, uri);
        intent.putExtra("outputFormat", Bitmap.CompressFormat.PNG.toString());

        startActivityForResult(intent, Config.REQUEST_CHOP_PHOTO_CODE);
    }

    private boolean initFile() {
        String filePath = mBufferPath + System.currentTimeMillis() + ".png";
        imageFile = new File(filePath);
        if (!imageFile.exists()) {
            try {
                return imageFile.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
                return false;
            }
        }
        return true;
    }

    // decide if the runtime permission is needed
    private boolean canMakeSmores() {
        return Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP_MR1;
    }

    private void askForPermission() {
        if (canMakeSmores()) {
            String[] perms = {
                    "android.permission.CAMERA",
                    "android.permission.WRITE_EXTERNAL_STORAGE",
                    "android.permission.READ_EXTERNAL_STORAGE"
            };
            int permsRequestCode = 200;
            requestPermissions(perms, permsRequestCode);
        }
    }

    private void readLabels() {
        mLabels = new String[1000];
        File f = new File(mModelPath + "labels.txt");
        Scanner s;
        int iter = 0;

        try {
            s = new Scanner(f);
            while (s.hasNextLine()) {
                String str = s.nextLine();
                mLabels[iter++] = str;
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
}
