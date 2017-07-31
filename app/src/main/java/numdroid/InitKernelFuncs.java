package numdroid;

import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.Type;
import android.util.Log;

import layers.ScriptC_convRolledInF4OutF1;
import layers.ScriptC_convRolledInF4OutF2;
import layers.ScriptC_convRolledInF4OutF4;
import layers.ScriptC_convRolledInF4OutF8;
import layers.ScriptC_convRolledInF8OutF1;
import layers.ScriptC_convRolledInF8OutF2;
import layers.ScriptC_convRolledInF8OutF4;
import layers.ScriptC_convRolledInF8OutF8;

/**
 * Created by Administrator on 2017/7/30 0030.
 */

public class InitKernelFuncs {
    private ScriptC_convRolledInF4OutF1 myScript41=null;
    private ScriptC_convRolledInF4OutF2 myScript42=null;
    private ScriptC_convRolledInF4OutF4 myScript44=null;
    private ScriptC_convRolledInF4OutF8 myScript48=null;
    private ScriptC_convRolledInF8OutF1 myScript81=null;
    private ScriptC_convRolledInF8OutF2 myScript82=null;
    private ScriptC_convRolledInF8OutF4 myScript84=null;
    private ScriptC_convRolledInF8OutF8 myScript88=null;
    RenderScript myRS;
    int group;
    int[] stride;
    int[] pad;
    public InitKernelFuncs(RenderScript myRS,int group,int[] stride,int[] pad)
    {
        this.myRS=myRS;
        this.group=group;
        this.stride=stride;
        this.pad=pad;
    }

    public void initKernelFrFs(float[][][][] myWeight, float[] myBias,int r,int s) {
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;

        int c_k_r = c_k;
        if (c_k % r != 0)
            c_k_r = c_k + r - c_k % r;

        int n_k_s = n_k;
        if (n_k % s != 0)
            n_k_s = n_k + s - n_k % s;

        Log.d("initkernel",String.valueOf(n_k));
        Allocation kernelAllocation;
        Allocation biasAllocation;
        Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_s * c_k_r * h_k * w_k / 4);
        Type biasType=Type.createX(myRS, Element.F32(myRS), n_k_s);
        switch(s)
        {
            case 1:
                break;
            case 2:
                biasType = Type.createX(myRS, Element.F32_2(myRS), n_k_s/2);
                break;
            case 4:
            case 8:
                biasType = Type.createX(myRS, Element.F32_4(myRS), n_k_s/4);
                break;
        }
        //Type kernelType = Type.createX(myRS, Element.F32_4(myRS), n_k_s * c_k_r * h_k * w_k / 4);
        //Type biasType = Type.createX(myRS, Element.F32_2(myRS), n_k_2 / 2);

        float[] kernelMatrix = new float[n_k_s * h_k * w_k * c_k_r];
        float[] biasArray = new float[n_k_s];
        int delta_n = (n_k_s - n_k) / group;
        for (int i = 0; i < n_k_s; i++)
            for (int j = 0; j < c_k_r; j++)
                for (int k = 0; k < h_k; k++)
                    for (int l = 0; l < w_k; l++) {
                        if (j >= c_k || ((i >= n_k_s / group - delta_n) && (i < n_k_s / group)) || (i >= n_k_s - delta_n))
                            kernelMatrix[i * h_k * w_k * c_k_r + k * w_k * c_k_r + l * c_k_r + j] = 0;
                        else if (i >= n_k_s / group)
                            kernelMatrix[i * h_k * w_k * c_k_r + k * w_k * c_k_r + l * c_k_r + j] = myWeight[i - delta_n][j][k][l];
                        else
                            kernelMatrix[i * h_k * w_k * c_k_r + k * w_k * c_k_r + l * c_k_r + j] = myWeight[i][j][k][l];
                    }

        for (int i = 0; i < n_k_s; i++) {
            if (((i >= n_k_s / group - delta_n) && (i < n_k_s / group)) || (i >= n_k_s - delta_n))
                biasArray[i] = 0;
            else if (i >= n_k_s / group)
                biasArray[i] = myBias[i - delta_n];
            else
                biasArray[i] = myBias[i];
        }
        kernelAllocation = Allocation.createTyped(myRS, kernelType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        kernelAllocation.copyFrom(kernelMatrix);

        biasAllocation = Allocation.createTyped(myRS, biasType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);
        biasAllocation.copyFrom(biasArray);
        if(r==4)
        {
            if(s==1)
            {
                myScript41 = new ScriptC_convRolledInF4OutF1(myRS);
                myScript41.set_Bias_Blob(biasAllocation);
                myScript41.set_Kernel_Blob(kernelAllocation);
                myScript41.set_n_k(n_k_s);
                myScript41.set_c_k(c_k_r);
                myScript41.set_h_k(h_k);
                myScript41.set_w_k(w_k);
                myScript41.set_pad_x(pad[0]);
                myScript41.set_pad_y(pad[1]);
                myScript41.set_stride_x(stride[0]);
                myScript41.set_stride_y(stride[1]);
                myScript41.set_group(group);
            }
            if(s==2)
            {
                myScript42 = new ScriptC_convRolledInF4OutF2(myRS);
                myScript42.set_Bias_Blob(biasAllocation);
                myScript42.set_Kernel_Blob(kernelAllocation);
                myScript42.set_n_k(n_k_s);
                myScript42.set_c_k(c_k_r);
                myScript42.set_h_k(h_k);
                myScript42.set_w_k(w_k);
                myScript42.set_pad_x(pad[0]);
                myScript42.set_pad_y(pad[1]);
                myScript42.set_stride_x(stride[0]);
                myScript42.set_stride_y(stride[1]);
                myScript42.set_group(group);
            }
            if(s==4)
            {
                myScript44 = new ScriptC_convRolledInF4OutF4(myRS);
                myScript44.set_Bias_Blob(biasAllocation);
                myScript44.set_Kernel_Blob(kernelAllocation);
                myScript44.set_n_k(n_k_s);
                myScript44.set_c_k(c_k_r);
                myScript44.set_h_k(h_k);
                myScript44.set_w_k(w_k);
                myScript44.set_pad_x(pad[0]);
                myScript44.set_pad_y(pad[1]);
                myScript44.set_stride_x(stride[0]);
                myScript44.set_stride_y(stride[1]);
                myScript44.set_group(group);
            }
            if(s==8)
            {
                myScript48 = new ScriptC_convRolledInF4OutF8(myRS);
                myScript48.set_Bias_Blob(biasAllocation);
                myScript48.set_Kernel_Blob(kernelAllocation);
                myScript48.set_n_k(n_k_s);
                myScript48.set_c_k(c_k_r);
                myScript48.set_h_k(h_k);
                myScript48.set_w_k(w_k);
                myScript48.set_pad_x(pad[0]);
                myScript48.set_pad_y(pad[1]);
                myScript48.set_stride_x(stride[0]);
                myScript48.set_stride_y(stride[1]);
                myScript48.set_group(group);
            }
        }
        else if(r==8)
        {
            if(s==1)
            {
                myScript81 = new ScriptC_convRolledInF8OutF1(myRS);
                myScript81.set_Bias_Blob(biasAllocation);
                myScript81.set_Kernel_Blob(kernelAllocation);
                myScript81.set_n_k(n_k_s);
                myScript81.set_c_k(c_k_r);
                myScript81.set_h_k(h_k);
                myScript81.set_w_k(w_k);
                myScript81.set_pad_x(pad[0]);
                myScript81.set_pad_y(pad[1]);
                myScript81.set_stride_x(stride[0]);
                myScript81.set_stride_y(stride[1]);
                myScript81.set_group(group);
            }
            if(s==2)
            {
                myScript82 = new ScriptC_convRolledInF8OutF2(myRS);
                myScript82.set_Bias_Blob(biasAllocation);
                myScript82.set_Kernel_Blob(kernelAllocation);
                myScript82.set_n_k(n_k_s);
                myScript82.set_c_k(c_k_r);
                myScript82.set_h_k(h_k);
                myScript82.set_w_k(w_k);
                myScript82.set_pad_x(pad[0]);
                myScript82.set_pad_y(pad[1]);
                myScript82.set_stride_x(stride[0]);
                myScript82.set_stride_y(stride[1]);
                myScript82.set_group(group);
            }
            if(s==4)
            {
                myScript84 = new ScriptC_convRolledInF8OutF4(myRS);
                myScript84.set_Bias_Blob(biasAllocation);
                myScript84.set_Kernel_Blob(kernelAllocation);
                myScript84.set_n_k(n_k_s);
                myScript84.set_c_k(c_k_r);
                myScript84.set_h_k(h_k);
                myScript84.set_w_k(w_k);
                myScript84.set_pad_x(pad[0]);
                myScript84.set_pad_y(pad[1]);
                myScript84.set_stride_x(stride[0]);
                myScript84.set_stride_y(stride[1]);
                myScript84.set_group(group);
            }
            if(s==8)
            {
                myScript88 = new ScriptC_convRolledInF8OutF8(myRS);
                myScript88.set_Bias_Blob(biasAllocation);
                myScript88.set_Kernel_Blob(kernelAllocation);
                myScript88.set_n_k(n_k_s);
                myScript88.set_c_k(c_k_r);
                myScript88.set_h_k(h_k);
                myScript88.set_w_k(w_k);
                myScript88.set_pad_x(pad[0]);
                myScript88.set_pad_y(pad[1]);
                myScript88.set_stride_x(stride[0]);
                myScript88.set_stride_y(stride[1]);
                myScript88.set_group(group);
            }
        }
        Log.d("initkernel","&&&&&&&&&&initkernel end");
    }
    public float[][][][] convLayerRolledParInFrOutFs(float[][][][] inputBlob, float[][][][] myWeight, boolean destroy,int r,int s,boolean nonLinear) {
        /*
        Convolution layer.
        Inputs:
        kernel[0] is a filter blob
        kernel[1] is bias blob
        */

        // calculate sizes
        //(n_i, c_i, h_i, w_i) = inputBlob.shape
        Log.d("initkernel","&&&&&&&&&&convLayerRolledParInFrOutFs");
        int n_i = inputBlob.length;
        int c_i = inputBlob[0].length;
        int h_i = inputBlob[0][0].length;
        int w_i = inputBlob[0][0][0].length;

        //(n_k, c_k, h_k, w_k) = kernel_blob[0].shape
        int n_k = myWeight.length;
        int c_k = myWeight[0].length;
        int h_k = myWeight[0][0].length;
        int w_k = myWeight[0][0][0].length;


        int n_o = n_i;
        int h_o = (int) (Math.ceil((h_i + 2 * pad[0] - h_k) / ((float) (stride[0]))) + 1);
        int w_o = (int) (Math.ceil((w_i + 2 * pad[1] - w_k) / ((float) (stride[1]))) + 1);
        int c_o = n_k;

        // initialize the result
        float[][][][] outputBlob = new float[n_o][c_o][h_o][w_o];

        int c_i_r = c_i;
        if (c_i % (r * group) != 0)
            c_i_r= c_i + (r * group) - c_i % (r * group);

        int n_k_s = n_k;
        if (n_k % (s * group) != 0)
            n_k_s = n_k + (s * group) - n_k % (s * group);

        int delta_n = (n_k_s - n_k) / group;

        //initialize Renderscript
        Type inputType;

        Allocation frameAllocation;
        Allocation outAllocation;
        inputType = Type.createX(myRS, Element.F32_4(myRS), c_i_r * h_i * w_i / 4);
        Type outType=Type.createX(myRS, Element.F32(myRS), h_o * w_o * n_k_s);
        switch(s)
        {
            case 1:
            break;
            case 2:
                outType=Type.createX(myRS, Element.F32_2(myRS), h_o * w_o * n_k_s/2);
            break;
            case 4:
            case 8:
                outType=Type.createX(myRS, Element.F32_4(myRS), h_o * w_o * n_k_s/4);
            break;
        }

        Log.d("initkernel","&&&&&&&&&& middle convLayerRolledParInFrOutFs");
        Log.d("initkernel",String.valueOf(n_k_s));
       // outType = Type.createX(myRS, Element.F32_2(myRS), h_o * w_o * n_k_2 / 2);

        frameAllocation = Allocation.createTyped(myRS, inputType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        outAllocation = Allocation.createTyped(myRS, outType, Allocation.MipmapControl.MIPMAP_NONE, Allocation.USAGE_GRAPHICS_TEXTURE | Allocation.USAGE_SCRIPT);

        if(r==4)
        {
            if(s==1)
            {
                myScript41.set_c_i(c_i_r);
                myScript41.set_h_i(h_i);
                myScript41.set_w_i(w_i);
                myScript41.set_h_o(h_o);
                myScript41.set_w_o(w_o);
            }
            if(s==2)
            {
                myScript42.set_c_i(c_i_r);
                myScript42.set_h_i(h_i);
                myScript42.set_w_i(w_i);
                myScript42.set_h_o(h_o);
                myScript42.set_w_o(w_o);
            }
            if(s==4)
            {
                myScript44.set_c_i(c_i_r);
                myScript44.set_h_i(h_i);
                myScript44.set_w_i(w_i);
                myScript44.set_h_o(h_o);
                myScript44.set_w_o(w_o);
            }
            if(s==8)
            {
                myScript48.set_c_i(c_i_r);
                myScript48.set_h_i(h_i);
                myScript48.set_w_i(w_i);
                myScript48.set_h_o(h_o);
                myScript48.set_w_o(w_o);
            }
        }
        else if(r==8)
        {
            if(s==1)
            {
                myScript81.set_c_i(c_i_r);
                myScript81.set_h_i(h_i);
                myScript81.set_w_i(w_i);
                myScript81.set_h_o(h_o);
                myScript81.set_w_o(w_o);
            }
            if(s==2)
            {
                myScript82.set_c_i(c_i_r);
                myScript82.set_h_i(h_i);
                myScript82.set_w_i(w_i);
                myScript82.set_h_o(h_o);
                myScript82.set_w_o(w_o);
            }
            if(s==4)
            {
                myScript84.set_c_i(c_i_r);
                myScript84.set_h_i(h_i);
                myScript84.set_w_i(w_i);
                myScript84.set_h_o(h_o);
                myScript84.set_w_o(w_o);
            }
            if(s==8)
            {
                myScript88.set_c_i(c_i_r);
                myScript88.set_h_i(h_i);
                myScript88.set_w_i(w_i);
                myScript88.set_h_o(h_o);
                myScript88.set_w_o(w_o);
            }
        }


        // calculate the result
        float[] outMatrix = new float[h_o * w_o * n_k_s];
        float[] frameMatrix = new float[h_i * w_i * c_i_r];
        int delta_c = (c_i_r - c_i) / group;

        for (int n = 0; n < (n_i); n++) {// for n in images
            if (n == 0) {
                for (int i = 0; i < c_i_r; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_r / group - delta_c) && (i < c_i_r/ group)) || (i >= c_i_r - delta_c))
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = 0;
                            else if (i >= c_i_r / group)
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = inputBlob[n][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = inputBlob[n][i][j][k];
                        }
            }
            frameAllocation.copyFrom(frameMatrix);
            if(r==4)
            {
                if(s==1)
                {
                    myScript41.set_In_Blob(frameAllocation);
                    myScript41.forEach_root(outAllocation);
                }
                if(s==2)
                {
                    myScript42.set_In_Blob(frameAllocation);
                    myScript42.forEach_root(outAllocation);
                }
                if(s==4)
                {
                    myScript44.set_In_Blob(frameAllocation);
                    myScript44.forEach_root(outAllocation);
                }
                if(s==8)
                {
                    myScript48.set_In_Blob(frameAllocation);
                    myScript48.forEach_root(outAllocation);
                }
            }
            else if(r==8)
            {
                if(s==1)
                {
                    myScript81.set_In_Blob(frameAllocation);
                    myScript81.forEach_root(outAllocation);
                }
                if(s==2)
                {
                    myScript82.set_In_Blob(frameAllocation);
                    myScript82.forEach_root(outAllocation);
                }
                if(s==4)
                {
                    myScript84.set_In_Blob(frameAllocation);
                    myScript84.forEach_root(outAllocation);
                }
                if(s==8)
                {
                    myScript88.set_In_Blob(frameAllocation);
                    myScript88.forEach_root(outAllocation);
                }
            }


            if (n < n_i - 1) {
                for (int i = 0; i < c_i_r; i++)
                    for (int j = 0; j < h_i; j++)
                        for (int k = 0; k < w_i; k++) {
                            if (((i >= c_i_r / group - delta_c) && (i < c_i_r/ group)) || (i >= c_i_r - delta_c))
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = 0;
                            else if (i >= c_i_r / group)
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = inputBlob[n + 1][i - delta_c][j][k];
                            else
                                frameMatrix[j * w_i * c_i_r + k * c_i_r + i] = inputBlob[n + 1][i][j][k];
                        }
            }

            if (n > 0) {
                for (int i = 0; i < n_k_s; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_s / group - delta_n) {
                                outputBlob[n - 1][i][j][k] = outMatrix[j * w_o * n_k_s + k * n_k_s + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i][j][k] < 0)
                                        outputBlob[n - 1][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_s / group) && (i < n_k_s - delta_n)) {
                                outputBlob[n - 1][i - delta_n][j][k] = outMatrix[j * w_o * n_k_s + k * n_k_s + i];
                                if (nonLinear) {
                                    if (outputBlob[n - 1][i - delta_n][j][k] < 0)
                                        outputBlob[n - 1][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }

            outAllocation.copyTo(outMatrix);

            if (n == n_i - 1) {
                for (int i = 0; i < n_k_s; i++)
                    for (int j = 0; j < h_o; j++)
                        for (int k = 0; k < w_o; k++) {
                            if (i < n_k_s / group - delta_n) {
                                outputBlob[n][i][j][k] = outMatrix[j * w_o * n_k_s + k * n_k_s + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i][j][k] < 0)
                                        outputBlob[n][i][j][k] = 0;
                                }
                            } else if ((i >= n_k_s / group) && (i < n_k_s - delta_n)) {
                                outputBlob[n][i - delta_n][j][k] = outMatrix[j * w_o * n_k_s + k * n_k_s + i];
                                if (nonLinear) {
                                    if (outputBlob[n][i - delta_n][j][k] < 0)
                                        outputBlob[n][i - delta_n][j][k] = 0;
                                }
                            }
                        }
            }
        }

        frameAllocation.destroy();
        outAllocation.destroy();

        inputType.destroy();
        outType.destroy();

        if (destroy)
        {
            if(r==4)
            {
                switch(s)
                {
                    case 1:
                        myScript41.destroy();
                        break;
                    case 2:
                        myScript42.destroy();
                        break;
                    case 4:
                        myScript44.destroy();
                        break;
                    case 8:
                        myScript48.destroy();
                        break;
                }
            }
            else
            {
                switch(s)
                {
                    case 1:
                        myScript81.destroy();
                        break;
                    case 2:
                        myScript82.destroy();
                        break;
                    case 4:
                        myScript84.destroy();
                        break;
                    case 8:
                        myScript88.destroy();
                        break;
                }
            }
        }
        Log.d("initkernel","&&&&&&&&&& end convLayerRolledParInFrOutFs");
        return outputBlob;

    }

}
