package com.redwhale.socialeyesandroid;

import android.graphics.Bitmap;
import java.util.Arrays;


public class ImageProc {
    public static Bitmap color(Bitmap srcimg) {
        float[][][] img = bitmap_to_array(srcimg);
        float[][][] img_color = enhance_color(img);
        return array_to_bitmap(img_color);
    }

    public static Bitmap redfree(Bitmap srcimg) {
        float[][][] img = bitmap_to_array(srcimg);
        float[][][] img_color = enhance_color(img);
        float[][] img_redfree = enhance_nored(img_color);
        return grayscale_array_to_bitmap(img_redfree);
    }

    public static Bitmap tone(Bitmap srcimg, float gamma) {
        float[][][] img = bitmap_to_array(srcimg);
        float[][][] img_color = enhance_color(img);
        float[][] img_redfree = enhance_nored(img_color);
        float[][] img_tone = enhance_tone(img_redfree, gamma);
        return grayscale_array_to_bitmap(img_tone);
    }

    public static Bitmap sharpen(Bitmap srcimg, float sigma) {
        float[][][] img = bitmap_to_array(srcimg);
        float[][][] img_color = enhance_color(img);
        float[][] img_redfree = enhance_nored(img_color);
        float[][] img_sharpen = enhance_sharpen(img_redfree, 0.5f, sigma);
        return grayscale_array_to_bitmap(img_sharpen);
    }

    public static Bitmap normalize(Bitmap srcimg) {
        float[][] img = image_to_array_green(srcimg);

        // xc, yc and r are the center and radius of field of view (the round area which is not black)
        // if not known, can be set as below (with somewhat less than perfect results), or set r=0 to disable this stage of the processing
        int r = (int)(srcimg.getHeight() * 0.8 / 2);
        int yc = srcimg.getHeight() / 2;
        int xc = srcimg.getWidth() / 2;

        // the scale of variation of the image should be set to 10-12% of the field of view
        int s = (int)(srcimg.getHeight() * 0.1);

        float[][] img_norm = enhance_normalize(img, s, yc, xc, r);

        return grayscale_array_to_bitmap(img_norm);
    }

    // NOTE this can be sped up with Bitmap.getPixels()
    public static float[][][] bitmap_to_array(Bitmap bimg) {
        int rows = bimg.getHeight();
        int cols = bimg.getWidth();
        int channels = 3; // exactly 3 channels is required
        float[][][] img = new float[rows][cols][channels];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int p = bimg.getPixel(j, i);
                img[i][j][0] = ((p >> 16) & 0xFF) / 255.0f;
                img[i][j][1] = ((p >>  8) & 0xFF) / 255.0f;
                img[i][j][2] = ((p >>  0) & 0xFF) / 255.0f;
            }
        }

        return img;
    }

    // NOTE this can be sped up with Bitmap.createBitmap (int[] colors, ...)
    public static Bitmap array_to_bitmap(float[][][] img) {
        int rows = img.length;
        int cols = img[0].length;
        Bitmap bimg = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float r = img[i][j][0];
                float g = img[i][j][1];
                float b = img[i][j][2];

                int ri = Math.max(Math.min( (int)Math.round(255*r), 255), 0);
                int gi = Math.max(Math.min( (int)Math.round(255*g), 255), 0);
                int bi = Math.max(Math.min( (int)Math.round(255*b), 255), 0);

                int p = (0xFF << 24) | (ri << 16) | (gi << 8) | (bi << 0);
                bimg.setPixel( j, i, p );
            }
        }

        return bimg;
    }

    private static float[][] image_to_array_green(Bitmap image) {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][] result = new float[height][width];

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                // pull out green from value
                result[row][col] = ((image.getPixel(col, row) >> 8) & 0xFF) / 255.0f;
            }
        }

        return result;
    }

    public static Bitmap grayscale_array_to_bitmap(float[][] img) {
        int rows = img.length;
        int cols = img[0].length;
        Bitmap bimg = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float gray = img[i][j];

                int gray_i = Math.max(Math.min( (int)Math.round(255*gray), 255), 0);

                // TODO there is probably a better conversion here, but this is close
                int p = (0xFF << 24) | (gray_i << 16) | (gray_i << 8) | (gray_i << 0);

                bimg.setPixel( j, i, p );
            }
        }

        return bimg;
    }

    public static float[][][] enhance_color(float[][][] img)
    {
        int rows = img.length;
        int cols = img[0].length;
        int channels = img[0][0].length; // must be 3
        float[][][] lab_img = new float[rows][cols][channels];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                rgb_to_lab(img[i][j], lab_img[i][j]);
            }
        }

        // grab the 1st and 99th percentile
        // can use something like http://commons.apache.org/proper/commons-math/javadocs/api-3.0/org/apache/commons/math3/stat/descriptive/rank/Percentile.html
        float[] l_values = new float[rows*cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                l_values[ i*cols + j ] = lab_img[i][j][0];
            }
        }
        Arrays.sort(l_values);
        float l_min = l_values[ (int)Math.round( 0.01 * (l_values.length-1) ) ];
        float l_max = l_values[ (int)Math.round( 0.99 * (l_values.length-1) ) ];
        System.out.println("L min " + l_min + " max " + l_max);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float l = lab_img[i][j][0];
                l = 100 * (l - l_min) / (l_max - l_min);
                lab_img[i][j][0] = l; // Math.max( Math.min( l, 255), 0 );
            }
        }

        float[][][] out_img = new float[rows][cols][channels];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                lab_to_rgb(lab_img[i][j], out_img[i][j]);
            }
        }

        return out_img;
    }

    public static float[][] enhance_nored(float[][][] img) {
        int rows = img.length;
        int cols = img[0].length;
        float[][] out_img = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Coefficients from: http://www.poynton.com/notes/colour_and_gamma/ColorFAQ.html#RTFToC9
                out_img[i][j] = 0.7152f * img[i][j][1] + 0.0722f * img[i][j][2];
            }
        }

        return out_img;
    }

    public static float[][] enhance_tone(float[][] img, float gamma_value) {
        int rows = img.length;
        int cols = img[0].length;
        float[][] out_img = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out_img[i][j] = (float)Math.pow(img[i][j], gamma_value);
            }
        }

        return out_img;
    }

    public static float[][] enhance_sharpen(float[][] img, float sharpen_value, float sigma) {
        int rows = img.length;
        int cols = img[0].length;
        float[][] out_img = new float[rows][cols];

        float[][] tmp_img = gaussian_blur_2d(img, sigma);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out_img[i][j] = img[i][j] * (1 + sharpen_value) - tmp_img[i][j] * sharpen_value;
            }
        }

        return out_img;
    }

    public static float[][] enhance_normalize(float[][] img, int s, int yc, int xc, int r) {
        int M = img.length;
        int N = img[0].length;

        byte[][] mask = new byte[M][N];
        for (int i = 0; i < M; i++) {
            Arrays.fill(mask[i], (byte)1);
        }
        if (r != 0) {
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float d = (float)Math.sqrt( (i-yc)*(i-yc) + (j-xc)*(j-xc) );
                    if (d > r) {
                        mask[i][j] = 0;
                    }
                    else {
                        mask[i][j] = 1;
                    }
                }
            }
        }

        float[][] img_median = median_filter(img, s, mask);

        if (r != 0) {
            extrapolate_nearest_outside_circle(img_median, yc, xc, r);
        }

        float[][] output = new float[M][N];

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                output[i][j] = ( img[i][j] / img_median[i][j] ) - 0.5f;
                if (output[i][j] < 0f) { output[i][j] = 0f; }
                if (output[i][j] > 1f) { output[i][j] = 1f; }
            }
        }

        return output;
    }

    public static float[][] extrapolate_nearest_outside_circle(float[][] img, int yc, int xc, int r) {
        int M = img.length;
        int N = img[0].length;

        // runs in-place for speed/memory efficiency
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float d = (float)Math.sqrt( (i-yc)*(i-yc) + (j-xc)*(j-xc) );
                if (d > r) {
                    int i1 = (int)( yc + (i-yc)*(r/d) );
                    int j1 = (int)( xc + (j-xc)*(r/d) );
                    img[i][j] = img[i1][j1];
                }
            }
        }

        return img;
    }


    public static float[][] median_filter(float[][] img, int r, byte[][] mask) {
        int M = img.length;
        int N = img[0].length;

        float[][] output = new float[M][N];
        int nlevels = 1024;

        int[] histogram = new int[nlevels+1];
        for (int j = 0; j < N; j++) {
            int j1 = Math.max(0, j-r);
            int j2 = Math.min(N, j+r);

            Arrays.fill(histogram, 0);
            int count = 0;

            for (int i = 0; i < M; i++) {
                // update histogram
                if (i - r >= 0) {
                    for (int jc = j1; jc < j2; jc++) {
                        if (mask[i-r][jc] == 0) {
                            continue;
                        }
                        int v = (int) (img[i-r][jc]*nlevels);
                        histogram[v]--;
                        count --;
                    }
                }

                if (i + r < M) {
                    for (int jc = j1; jc < j2; jc++) {
                        if (mask[i+r][jc] == 0) {
                            continue;
                        }
                        int v = (int) (img[i+r][jc]*nlevels);
                        histogram[ v ] ++;
                        count ++;
                    }
                }

                // find median in histogram
                int sum = 0;
                for (int k = 0; k < histogram.length; k++) {
                    sum += histogram[k];
                    if (sum >= count / 2.0f) {
                        // this version picks a quasi-median between k and k-1 weighted based on the size of the step relative to the position of the exact median value
                        float w = (sum - (count / 2.0f)) / histogram[k];
                        output[i][j] = (k * (1-w) + (k-1) * w) / nlevels;

                        // this is a simple histogram
                        // output[i][j] = ((float)k) / nlevels;
                        break;
                    }
                }
            }
        }

        return output;
    }

    public static float[][] enhance_deconvolve(float[][] img) {
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

    public static float[][] convolve_2d(float[][] img, float[][] kernel) {
        int M = img.length;
        int N = img[0].length;
        int KM = kernel.length;
        int KN = kernel[0].length;
        int KM2 = KM/2;
        int KN2 = KN/2;

        float[][] output = new float[M][N];

        // TODO handle edges better: same options as scipy.ndimage.convolve
        // the current code reflects at edges, but is about 3x slower than code which ignores edges
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                for (int ki = 0; ki < KM; ki++) {
                    for (int kj = 0; kj < KN; kj++) {
                        int i1 = i+ki-KM2;
                        int j1 = j+kj-KN2;
                        if (i1 < 0) {
                            i1 = -i1;
                        } else if (i1 >= M) {
                            i1 = -i1 + 2*M - 1;
                        }
                        if (j1 < 0) {
                            j1 = -j1;
                        } else if (j1 >= N) {
                            j1 = -j1 + 2*N - 1;
                        }

                        output[i][j] += kernel[ki][kj] * img[i1][j1];
                    }
                }
            }
        }
        return output;
    }

    public static float[][] gaussian_blur_2d(float[][] img, float sigma) {
        int K = (int)Math.ceil(6*sigma);
        float[][] kernel_h = new float[K][1];
        float[][] kernel_v = new float[1][K];
        float sum = 0;
        for (int k = 0; k < K; k++) {
            kernel_h[k][0] = (float)Math.exp( - Math.pow( k - Math.round(K/2.0f), 2 ) / (2*sigma*sigma) );
            kernel_v[0][k] = kernel_h[k][0];
            sum += kernel_h[k][0];
        }
        for (int k = 0; k < K; k++) {
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
    public static void rgb_to_lab(float[] rgb, float[] lab) {
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

    public static void lab_to_rgb(float[] lab, float[] rgb) {
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