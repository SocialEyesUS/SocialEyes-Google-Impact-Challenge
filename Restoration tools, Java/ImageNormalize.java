/*
Normalize retinal images
Author: Alex Izvorski, August-September 2015
*/

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

import java.util.Arrays;
import java.io.IOException;
import java.io.File;

// import ImageUtils;

public class ImageNormalize
{

    public static void main(String[] args) throws IOException
    {
        String input_filename = args[0];
        String output_filename = args[1];
        BufferedImage bimg = ImageIO.read(new File(input_filename));
        float[][] img = ImageUtils.buffered_image_to_array_green(bimg);

        // xc, yc and r are the center and radius of field of view (the round area which is not black)
        // if not known, can be set as below (with somewhat less than perfect results), or set r=0 to disable this stage of the processing
        int r = (int)(bimg.getHeight() * 0.8 / 2);
        int yc = bimg.getHeight() / 2;
        int xc = bimg.getWidth() / 2;

        // the scale of variation of the image should be set to 10-12% of the field of view
        int s = (int)(bimg.getHeight() * 0.1);

        long t1 = System.nanoTime();

        float[][] img_norm = normalize(img, s, yc, xc, r);

        long t2 = System.nanoTime();
        System.out.println(String.format("%d ms", (t2-t1) / 1000000));

        bimg = ImageUtils.grayscale_array_to_buffered_image(img_norm);
        ImageIO.write(bimg, "png", new File(output_filename));
    }


    public static float[][] normalize(float[][] img, int s, int yc, int xc, int r)
    {
        int M = img.length;
        int N = img[0].length;

        byte[][] mask = new byte[M][N];
        for (int i = 0; i < M; i++)
        {
            Arrays.fill(mask[i], (byte)1);
        }
        if (r != 0)
        {
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    float d = (float)Math.sqrt( (i-yc)*(i-yc) + (j-xc)*(j-xc) );
                    if (d > r) { mask[i][j] = 0; }
                    else { mask[i][j] = 1; }
                }
            }
        }


        float[][] img_median = median_filter(img, s, mask);

        if (r != 0)
        {
            extrapolate_nearest_outside_circle(img_median, yc, xc, r);
        }

        float[][] output = new float[M][N];

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                output[i][j] = ( img[i][j] / img_median[i][j] ) - 0.5f;
                if (output[i][j] < 0f) { output[i][j] = 0f; }
                if (output[i][j] > 1f) { output[i][j] = 1f; }
            }
        }

        return output;
    }

    public static float[][] extrapolate_nearest_outside_circle(float[][] img, int yc, int xc, int r)
    {
        int M = img.length;
        int N = img[0].length;

        // runs in-place for speed/memory efficiency
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float d = (float)Math.sqrt( (i-yc)*(i-yc) + (j-xc)*(j-xc) );
                if (d > r)
                {
                    int i1 = (int)( yc + (i-yc)*(r/d) );
                    int j1 = (int)( xc + (j-xc)*(r/d) );
                    img[i][j] = img[i1][j1];
                }
            }
        }

        return img;
    }


    // This algorithm: T. Huang, G. Yang, and G. Tang, “A Fast Two-Dimensional Median Filtering Algorithm,” 
    // IEEE Trans. Acoust., Speech, Signal Processing, vol. 27, no. 1, pp. 13-18, 1979

    // Faster (but significantly more complex) algorithm: Perreault, Simon, and Patrick Hébert. "Median filtering in constant time." 
    // Image Processing, IEEE Transactions on 16.9 (2007): 2389-2394.
    // https://nomis80.org/ctmf.pdf
    public static float[][] median_filter(float[][] img, int r, byte[][] mask)
    {
        int M = img.length;
        int N = img[0].length;

        float[][] output = new float[M][N];
        int nlevels = 1024;

        int[] histogram = new int[nlevels+1];
        for (int j = 0; j < N; j++)
        {
            int j1 = Math.max(0, j-r);
            int j2 = Math.min(N, j+r);

            Arrays.fill(histogram, 0);
            int count = 0;

            for (int i = 0; i < M; i++)
            {
                // update histogram
                if (i - r >= 0)
                {
                    for (int jc = j1; jc < j2; jc++)
                    {
                        if (mask[i-r][jc] == 0) { continue; }
                        int v = (int) (img[i-r][jc]*nlevels);
                        histogram[ v ] --;
                        count --;
                    }
                }

                if (i + r < M)
                {
                    for (int jc = j1; jc < j2; jc++)
                    {
                        if (mask[i+r][jc] == 0) { continue; }
                        int v = (int) (img[i+r][jc]*nlevels);
                        histogram[ v ] ++;
                        count ++;
                    }
                }

                // find median in histogram
                int sum = 0;
                for (int k = 0; k < histogram.length; k++)
                {
                    sum += histogram[k];
                    if (sum >= count / 2.0f)
                    {
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

}