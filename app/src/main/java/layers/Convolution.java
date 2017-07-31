package layers;

import android.renderscript.RenderScript;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.util.Scanner;

import messagepack.ParamUnpacker;
import numdroid.InitKernelFuncs;
import numdroid.MyNum;

public class Convolution implements LayerInterface {

    private String name;                    // name of the layer
    private String paramFilePath;           // name of the file which specifies the weights and biases
    private ParamUnpacker paramUnpacker;    // for extracting the wieghts and biases from the parameters file
    private int[] stride;                   // strides
    private int[] pad;                      // pads
    private int group;                      // number of groups
    private MyNum myNum;                    // for mathematical calculations
    private RenderScript myRS;              // RenderScript object
    private boolean nonLinear;              // Does a non-linear layer follow this layer?
    private NonLinearType nonLinearType;    // non-linearity type (if applicable)
    private boolean parallel;               // implementation method (parallel or sequential)
    private boolean loadParamsAtStart;      // if true, layer parameters will be loaded at the construction of network, otherwise the parameters will be loaded in run time
    private float[][][][] weight;           // weight parameter of network
    private float[] bias;                   // bias parameter of network
    private String tuningFolder;            // location to store online tuning results
    private boolean tuneNow;                // flag to weather execute tuning ro not
    private boolean tuneFunc;               // flag of optional tuning function
    private String algorithm;               // acceleration method
    private String[] names = {"F4F1", "F4F2", "F4F4", "F4F8", "F8F1", "F8F2", "F8F4", "F8F8"};

    private InitKernelFuncs initKernelFuncs;

    // types of non-linear layer that may be appended to this layer
    public enum NonLinearType {
        RectifiedLinearUnit,
        None
    }

    public Convolution(int[] stride, int[] pad, int group, String paramFilePath, boolean parallel, boolean loadParamsAtStart, boolean tuneFunc, RenderScript myRS, String name, String tuningFolder) {
        this.paramFilePath = paramFilePath;
        this.stride = stride;
        this.pad = pad;
        this.group = group;
        this.nonLinearType = NonLinearType.None;
        this.nonLinear = false;
        this.myRS = myRS;
        this.parallel = parallel;
        this.name = name;
        this.myNum = new MyNum();
        this.paramUnpacker = new ParamUnpacker();
        this.loadParamsAtStart = loadParamsAtStart;
        this.tuneFunc = tuneFunc;
        this.tuningFolder = tuningFolder;
        this.initKernelFuncs=new InitKernelFuncs(this.myRS,group,stride,pad);
        Log.d("***********chushihua","____________________________");
        tuneNow = false;
        File f = new File(tuningFolder + "/" + name + ".txt");
        try {
            Scanner s = new Scanner(f);
            algorithm = s.nextLine();
            if (corrupted(algorithm))
                tuneNow = true;
        } catch (FileNotFoundException e) {
            tuneNow = true;
        }

        if (!this.tuneFunc) {
            algorithm = "F8F4";
            tuneNow = false;
        }
        Log.d("-----------tuneNow",String.valueOf(tuneNow));
        Log.d("-----------parallel",String.valueOf(parallel));
        if (loadParamsAtStart && (!tuneNow || !parallel)) {
            long loadTime = System.currentTimeMillis();

            Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
            weight = (float[][][][]) objects[0];
            bias = (float[]) objects[1];

            loadTime = System.currentTimeMillis() - loadTime;

            long kernelTime = System.currentTimeMillis();
            Log.d("CNNdroid", "layers." + name + ": Parameters Load Time in Constructor = " + String.valueOf(loadTime));

            if (parallel) {//don't tune now
                int InFNum=Integer.parseInt(algorithm.substring(1,2));
                int outFNum=Integer.parseInt(algorithm.substring(3));
                Log.d("outFNUM****************",String.valueOf(outFNum));
                initKernelFuncs.initKernelFrFs(weight,bias,InFNum,outFNum);
                kernelTime = System.currentTimeMillis() - kernelTime;
                Log.d("CNNdroid", "layers." + name + ": Kernel Initialization Time in Constructor = " + String.valueOf(kernelTime));
            }
        }
    }

    public void setNonLinearType(NonLinearType nonLinearType) {
        this.nonLinearType = nonLinearType;
        nonLinear = true;
    }

    @Override
    public Object compute(Object input) {
            return invokeFunctions(input, weight, bias, true);
    }


    ///////////////////////////////////////Sequential///////////////////////////////////////////////
    private float[][][][] convLayerRolledSeq(float[][][][] inputBlob, float[][][][] filterBlob,
                                             float[] biasBlob, int[] pad, int[] stride, int group) {
        /*
        Convolution Layer
        Inputs:
        kernel[0] is a filter blob.
        kernel[1] is bias blob.
        group
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = filterBlob.length;
        int c_k = filterBlob[0].length;
        int h_k = filterBlob[0][0].length;
        int w_k = filterBlob[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        // calculate the result
        for (int n = 0; n < (n_i); n++) // for n in images
            for (int k = 0; k < (n_k / group); k++)// for k in kernels
                for (int g = 0; g < (group); g++) {
                    float[][][] convInFrame = new float[(c_i / group)][h_i][w_i];
                    float[][][] convInKernel;

                    int temp = g * c_i / group;
                    for (int i = g * c_i / group; i < (g + 1) * c_i / group; i++) // copy part of inputBlob
                        convInFrame[i - temp] = inputBlob[n][i];

                    convInKernel = filterBlob[g * n_k / group + k];       // copy

                    outputBlob[n][k + g * n_k / group] = convRolledSeq(convInFrame, convInKernel, biasBlob[g * n_k / group + k], pad, stride);
                }

        // return the result
        return outputBlob;
    }

    private float[][] convRolledSeq(float[][][] frames, float[][][] kernel, float bias,
                                    int[] pad, int[] stride) {
        // Calculate final dimensions.
        int c_i = frames.length;
        int h_i = frames[0].length;
        int w_i = frames[0][0].length;

        int c_k = kernel.length;
        int h_k = kernel[0].length;
        int w_k = kernel[0][0].length;

        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) stride[0])) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) stride[1])) + 1);

        int h_s = stride[0];
        int w_s = stride[1];

        float[][] out = new float[h_o][w_o];

        // Compute pixel values.
        for (int i = 0; i < h_o; ++i)
            for (int j = 0; j < w_o; ++j)
                out[i][j] = myNum.sum_conv(frames, kernel, i * h_s, j * w_s, pad[0], pad[1]) + bias;

        return out;
    }

    /////////////////////////////////////////Tuning Function////////////////////////////////////////
    private Object tuneFunction(float[][][][] input) {
        Log.d("CNNdroid", "layers." + name + ": Tuning process is starting...");
        long tuneTime = System.currentTimeMillis();

        Object[] objects = paramUnpacker.unpackerFunction(paramFilePath, new Class[]{float[][][][].class, float[].class});
        float[][][][] myWeight = (float[][][][]) objects[0];
        float[] myBias = (float[]) objects[1];
        tuneNow = false;
        long[] time = new long[]{0, 0, 0, 0};
        long temp;
        int c_i = input[0].length;
        float[][][][] tuneInput = new float[1][c_i][input[0][0].length][input[0][0][0].length];
        tuneInput[0] = input[0];
        if (c_i < 5) {
            for (int i = 0; i < 2; i++) {
                temp = System.currentTimeMillis();
                Log.d("initkernel","&&&&&&&&&&ready enter my initkernel");
                initKernelFuncs.initKernelFrFs(myWeight,myBias,4,1);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,4,1,nonLinear);
                time[0] += System.currentTimeMillis() - temp;
                Log.d("initkernel","&&&&&&&&&&firsttune end");
                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,4,2);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,4,2,nonLinear);
                time[1] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,4,4);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,4,4,nonLinear);
                time[2] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,4,8);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,4,8,nonLinear);
                time[3] += System.currentTimeMillis() - temp;
            }

            int min = 0;
            for (int i = 0; i < 4; i++)
                if (time[i] <= time[min])
                    min = i;

            algorithm = names[min];
        } else {
            for (int i = 0; i < 2; i++) {
                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,8,1);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,8,1,nonLinear);
                time[0] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,8,2);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,8,2,nonLinear);
                time[1] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,8,4);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,8,4,nonLinear);
                time[2] += System.currentTimeMillis() - temp;

                temp = System.currentTimeMillis();
                initKernelFuncs.initKernelFrFs(myWeight,myBias,8,8);
                initKernelFuncs.convLayerRolledParInFrOutFs(tuneInput,myWeight,true,8,8,nonLinear);
                time[3] += System.currentTimeMillis() - temp;
            }

            int min = 0;
            for (int i = 0; i < 4; i++)
                if (time[i] <= time[min])
                    min = i;

            algorithm = names[min + 4];
        }

        initKernelFuncs.initKernelFrFs(myWeight,myBias,4,8);
        Object output = initKernelFuncs.convLayerRolledParInFrOutFs(input,myWeight,true,4,8,nonLinear);

        writeFile(algorithm);
//        if(loadParamsAtStart) {
//            weight = myWeight;
//            bias = myBias;
//            int InFNum=Integer.parseInt(algorithm.substring(1,2));
//            int outFNum=Integer.parseInt(algorithm.substring(3));
//            initKernelFuncs.initKernelFrFs(weight,bias,InFNum,outFNum);
//        }
        tuneTime = System.currentTimeMillis() - tuneTime;
        Log.d("CNNdroid", "layers." + name + ": Tuning process finished in " + tuneTime + "ms.");
        tuneNow=false;
        return output;
    }

    ////////////////////////////////////////Local Functions/////////////////////////////////////////
    private Object invokeFunctions(Object input, float[][][][] myWeight, float[] myBias, boolean destroy)
    {
        Log.d("&&&&&&&&&&invoke","enter invokeFunc");
        Object output = null;
        long runTime = System.currentTimeMillis();

        if (!parallel)
            output = convLayerRolledSeq((float[][][][]) input, myWeight, myBias, pad, stride, group);
        else {
            if (tuneNow) {
                Log.d("&&&&&&&&&&begintune","enter invokeFunc");
                output =  tuneFunction((float[][][][]) input);//set the value of algorithm for next run
            }
            else {
                int InFNum=Integer.parseInt(algorithm.substring(1,2));
                int outFNum=Integer.parseInt(algorithm.substring(3));
                initKernelFuncs.initKernelFrFs(myWeight,myBias,InFNum,outFNum);
                output = initKernelFuncs.convLayerRolledParInFrOutFs((float[][][][]) input,myWeight,destroy,InFNum,outFNum,nonLinear);
            }
        }

        runTime = System.currentTimeMillis() - runTime;
        Log.d("CNNdroid", "layers." + name + ": Computation Run Time = " + String.valueOf(runTime));

        return output;
    }
    private boolean corrupted(String str)
    {
        for (int i = 0 ; i < names.length ; i++)
            if (str.equals(names[i]))
                return false;
        return true;
    }
    private void writeFile(String str)
    {
        File f = new File(tuningFolder + "/" + name + ".txt");

        if(f.exists())
            f.delete();
        try {
            f.createNewFile();
            FileOutputStream fos = new FileOutputStream(f);
            fos.write(str.getBytes());
            fos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

