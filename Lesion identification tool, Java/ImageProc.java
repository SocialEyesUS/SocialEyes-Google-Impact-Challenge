/*
Enhance retinal images
Author: Alex Izvorski, August 2015
Based on python code and algorithms by: Rick Morrison, Copyright 2015 Distant Focus Corporation
*/

// For desktop
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

// For Android
// import android.graphics.Bitmap;

import java.util.Arrays;
import java.io.IOException;
import java.io.File;

// import ImageUtils;

public class ImageProc
{

    public static void main(String[] args) throws IOException
    {
        String input_filename = args[0];
        //String output_filename = args[1];

        BufferedImage bimg = ImageIO.read(new File(input_filename));
        float[][][] img = ImageUtils.buffered_image_to_array(bimg);

        for (int i = 0; i < 10; i++)
        {
        long t1 = System.nanoTime();

        float[][][] img_color = enhance_color(img);
        bimg = ImageUtils.array_to_buffered_image(img_color);
        ImageIO.write(bimg, "png", new File("test_color.png"));

        float[][] img_nored = enhance_nored(img_color);
        bimg = ImageUtils.grayscale_array_to_buffered_image(img_nored);
        ImageIO.write(bimg, "png", new File("test_nored.png"));

        float[][] img_tone = enhance_tone(img_nored, 0.8f);
        bimg = ImageUtils.grayscale_array_to_buffered_image(img_tone);
        ImageIO.write(bimg, "png", new File("test_tone.png"));

        float[][] img_sharpen = enhance_sharpen(img_nored, 0.5f, 2f);
        bimg = ImageUtils.grayscale_array_to_buffered_image(img_sharpen);
        ImageIO.write(bimg, "png", new File("test_sharpen.png"));

        long t2 = System.nanoTime();
        System.out.println(String.format("%d ms", (t2-t1) / 1000000));
        }
    }

    public static float[][][] enhance_color(float[][][] img)
    {
        int channels = img.length; // must be 3
        int rows = img[0].length;
        int cols = img[0][0].length;
        float[][][] lab_img = new float[channels][rows][cols];

        float[] rgb = new float[3];
        float[] lab = new float[3];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                rgb[0] = img[0][i][j]; rgb[1] = img[1][i][j]; rgb[2] = img[2][i][j];
                rgb_to_lab(rgb, lab);
                lab_img[0][i][j] = lab[0]; lab_img[1][i][j] = lab[1]; lab_img[2][i][j] = lab[2]; 
            }
        }

        // grab the 1st and 99th percentile
        // can use something like http://commons.apache.org/proper/commons-math/javadocs/api-3.0/org/apache/commons/math3/stat/descriptive/rank/Percentile.html
        float[] l_values = new float[rows*cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                l_values[ i*cols + j ] = lab_img[0][i][j];
            }
        }
        Arrays.sort(l_values);
        float l_min = l_values[ (int)Math.round( 0.01 * (l_values.length-1) ) ];
        float l_max = l_values[ (int)Math.round( 0.99 * (l_values.length-1) ) ];
        System.out.println("L min " + l_min + " max " + l_max);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float l = lab_img[0][i][j];
                l = 100 * (l - l_min) / (l_max - l_min);
                lab_img[0][i][j] = l; // Math.max( Math.min( l, 255), 0 );
            }
        }

        float[][][] out_img = new float[channels][rows][cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                lab[0] = lab_img[0][i][j]; lab[1] = lab_img[1][i][j]; lab[2] = lab_img[2][i][j];
                lab_to_rgb(lab, rgb);
                out_img[0][i][j] = rgb[0]; out_img[1][i][j] = rgb[1]; out_img[2][i][j] = rgb[2];
            }
        }

        return out_img;
    }

    public static float[][] enhance_nored(float[][][] img)
    {
        int channels = img.length;
        int rows = img[0].length;
        int cols = img[0][0].length;
        float[][] out_img = new float[rows][cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Coefficients from: http://www.poynton.com/notes/colour_and_gamma/ColorFAQ.html#RTFToC9
                out_img[i][j] = 0.7152f * img[1][i][j] + 0.0722f * img[2][i][j];
            }
        }

        return out_img;
    }

    public static float[][] enhance_tone(float[][] img, float gamma_value)
    {
        int rows = img.length;
        int cols = img[0].length;
        float[][] out_img = new float[rows][cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                out_img[i][j] = (float)Math.pow(img[i][j], gamma_value);
            }
        }

        return out_img;
    }

    public static float[][] enhance_sharpen(float[][] img, float sharpen_value, float sigma)
    {
        int rows = img.length;
        int cols = img[0].length;
        float[][] out_img = new float[rows][cols];

        float[][] tmp_img = gaussian_blur_2d(img, sigma);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                out_img[i][j] = img[i][j] * (1 + sharpen_value) - tmp_img[i][j] * sharpen_value;
            }
        }

        return out_img;
    }

    public static float[][] enhance_deconvolve(float[][] img)
    {
        // TODO
        // DFT example

        /*
         import org.jtransforms.fft.FloatFFT_2D;
         ...
        int sz = 128;
        float[][] img = new float[sz][sz];
        FloatFFT_2D fft = new FloatFFT_2D(sz,sz);
        long t0 = System.nanoTime();
        fft.realForward(img);
        long t = System.nanoTime() - t0;
        System.out.println("Done " + t);
        */
        return null;
    }

    public static float[][] convolve_2d(float[][] img, float[][] kernel)
    {
        int M = img.length;
        int N = img[0].length;
        int KM = kernel.length;
        int KN = kernel[0].length;
        int KM2 = KM/2;
        int KN2 = KN/2;

        float[][] output = new float[M][N];

        // TODO handle edges better: same options as scipy.ndimage.convolve
        // the current code reflects at edges, but is about 3x slower than code which ignores edges
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int ki = 0; ki < KM; ki++)
                {
                    for (int kj = 0; kj < KN; kj++)
                    {
                        int i1 = i+ki-KM2;
                        int j1 = j+kj-KN2;
                        if (i1 < 0) { i1 = -i1; }
                        else if (i1 >= M) { i1 = -i1 + 2*M - 1; }
                        if (j1 < 0) { j1 = -j1; }
                        else if (j1 >= N) { j1 = -j1 + 2*N - 1; }

                        output[i][j] += kernel[ki][kj] * img[i1][j1];
                    }
                }
            }
        }
        return output;
    }

    public static float[][] gaussian_blur_2d(float[][] img, float sigma)
    {
        int K = (int)Math.ceil(6*sigma);
        float[][] kernel_h = new float[K][1];
        float[][] kernel_v = new float[1][K];
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            kernel_h[k][0] = (float)Math.exp( - Math.pow( k - Math.round(K/2.0f), 2 ) / (2*sigma*sigma) );
            kernel_v[0][k] = kernel_h[k][0];
            sum += kernel_h[k][0];
        }
        for (int k = 0; k < K; k++)
        {
            kernel_h[k][0] /= sum;
            kernel_v[0][k] /= sum;
        }
        float[][] tmp_img = convolve_2d(img, kernel_h);
        float[][] out_img = convolve_2d(tmp_img, kernel_v);
        return out_img;
    }

    /*
    The functions rgb_to_lab and lab_to_rgb are from the C3 package, https://github.com/StanfordHCI/c3/blob/master/java/src/edu/stanford/vis/color/LAB.java
    The originals are covered by C3's BSD-style license, https://github.com/StanfordHCI/c3/blob/master/LICENSE
    These versions have been modified to convert in-place and avoid any memory allocation (except for locals), and also to use float values instead of double
    */
    public static void rgb_to_lab(float[] rgb, float[] lab)
    {
        float r = rgb[0];
        float g = rgb[1];
        float b = rgb[2];

        // D65 standard referent
        float X = 0.950470f, Y = 1.0f, Z = 1.088830f;

        // second, map sRGB to CIE XYZ
        r = r <= 0.04045f ? r/12.92f : (float)Math.pow((r+0.055f)/1.055f, 2.4f);
        g = g <= 0.04045f ? g/12.92f : (float)Math.pow((g+0.055f)/1.055f, 2.4f);
        b = b <= 0.04045f ? b/12.92f : (float)Math.pow((b+0.055f)/1.055f, 2.4f);
        float x = (0.4124564f*r + 0.3575761f*g + 0.1804375f*b) / X,
              y = (0.2126729f*r + 0.7151522f*g + 0.0721750f*b) / Y,
              z = (0.0193339f*r + 0.1191920f*g + 0.9503041f*b) / Z;

        // third, map CIE XYZ to CIE L*a*b* and return
        x = x > 0.008856f ? (float)Math.pow(x, 1.0f/3) : 7.787037f*x + 4.0f/29;
        y = y > 0.008856f ? (float)Math.pow(y, 1.0f/3) : 7.787037f*y + 4.0f/29;
        z = z > 0.008856f ? (float)Math.pow(z, 1.0f/3) : 7.787037f*z + 4.0f/29;

        float L = 116*y - 16,
              A = 500*(x-y),
              B = 200*(y-z);

        lab[0] = L;
        lab[1] = A;
        lab[2] = B;
    }

    public static void lab_to_rgb(float[] lab, float[] rgb)
    {
        float L = lab[0];
        float A = lab[1];
        float B = lab[2];

        float y = (L + 16) / 116;
        float x = y + A/500;
        float z = y - B/200;

        // D65 standard referent
        float X = 0.950470f, Y = 1.0f, Z = 1.088830f;

        x = X * (x > 0.206893034f ? x*x*x : (x - 4.0f/29) / 7.787037f);
        y = Y * (y > 0.206893034f ? y*y*y : (y - 4.0f/29) / 7.787037f);
        z = Z * (z > 0.206893034f ? z*z*z : (z - 4.0f/29) / 7.787037f);

        // second, map CIE XYZ to sRGB
        float r =  3.2404542f*x - 1.5371385f*y - 0.4985314f*z;
        float g = -0.9692660f*x + 1.8760108f*y + 0.0415560f*z;
        float b =  0.0556434f*x - 0.2040259f*y + 1.0572252f*z;
        r = r <= 0.00304f ? 12.92f*r : 1.055f*(float)Math.pow(r,1/2.4f) - 0.055f;
        g = g <= 0.00304f ? 12.92f*g : 1.055f*(float)Math.pow(g,1/2.4f) - 0.055f;
        b = b <= 0.00304f ? 12.92f*b : 1.055f*(float)Math.pow(b,1/2.4f) - 0.055f;

        rgb[0] = r;
        rgb[1] = g;
        rgb[2] = b;
    }


}


/*
build and run with JTransforms on linux:

mkdir jars
cd jars ; unzip -x JTransforms-3.0.jar ; unzip -x JLargeArrays-1.2.jar ; cd ..
javac -cp "jars" ImageProc.java  ; cp ImageProc.class jars
java  -classpath 'jars' ImageProc
*/
