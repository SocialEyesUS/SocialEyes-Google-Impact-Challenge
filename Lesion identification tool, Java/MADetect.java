/*
Microaneurysm detector for retinal images - Java port
Author: Alex Izvorski, August 2015
*/

import java.lang.Math;
import java.util.ArrayDeque;
import java.awt.image.BufferedImage;
import java.awt.Point;
import java.awt.Graphics2D;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.io.File;

public class MADetect
{
    /*
    Demo code: read input image from disk, save output image to disk
    Compile: javac MADetect.java
    Run: java MADetect input.png output.png
    With the settings in here, the input is expected to be approx 600x600 to 800x800 in size
    */
    public static void main(String[] args) throws IOException
    {
        String input_filename = args[0];
        String output_filename = args[1];

        BufferedImage srcimg = ImageIO.read(new File(input_filename));
        System.out.println(String.format("Read image %s", input_filename));

        float[][] img = image_to_array_green(srcimg);

        float[][] kernel = make_kernel(17, 6, 1);
        float s_threshold = 0.26f;
        float d_threshold = -0.07f;
        float s_r = 12f;
        float d_r = 12f;
        float angle_step = 30f;

        long t1 = System.nanoTime();
        int[][] ma_image = ma_detect(img, kernel, s_threshold, d_threshold, s_r, d_r, angle_step);
        long t2 = System.nanoTime();

        System.out.println(String.format("MA detector done, %d ms", (t2-t1) / 1000000));

        // Make sure the output is color even if input is grayscale
        BufferedImage bimg = new BufferedImage(srcimg.getWidth(), srcimg.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bimg.createGraphics();
        g.drawImage(srcimg, 0, 0, null);
        // g.dispose();

        // Mark up identified MA areas
        int M = ma_image.length;
        int N = ma_image[0].length;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (ma_image[i][j] != 0)
                {
                    bimg.setRGB( i, j, bimg.getRGB(i, j) | (0xFF0000) );
                }
            }
        }

        ImageIO.write(bimg, "png", new File(output_filename));
        System.out.println(String.format("Wrote image %s", output_filename));

        /*
        // Test connected components
        int[][] test = 
            {
            {1,1,0,1,1,1,0,1}, 
            {1,1,0,1,0,1,0,1}, 
            {1,1,1,1,0,0,0,1}, 
            {0,0,0,0,0,0,0,1}, 
            {1,1,1,1,0,1,0,1}, 
            {0,0,0,1,0,1,0,1}, 
            {1,1,1,1,0,0,0,1}, 
            };
        int[][] labels = connected_components_2d(test);
        print_array_2d(labels);
        */

    }

    private static float[][] image_to_array_green(BufferedImage image)
    {
        int width = image.getWidth();
        int height = image.getHeight();
        float[][] result = new float[height][width];

        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                // pull out green from value
                result[row][col] = (image.getRGB(row, col) >> 8) & 0xFF;
            }
        }

        return result;
    }


    /*
    Create a circular kernel convolved with a gaussian
    A positive-negative offset pair of these is used in all the rest of the transforms
    */
    private static float[][] make_kernel(int size, float k_r, float sigma)
    {
        float[][] kernel = new float[size][size];
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float d = (float)Math.sqrt(Math.pow((x-size/2), 2) + Math.pow((y-size/2), 2));
                if (d < k_r)
                {
                    kernel[y][x] = 1;
                }
            }
        }
        // TODO
        // kernel = skimage.filters.gaussian_filter(kernel, sigma)
        return kernel;
    }


    /*
    Transform image into a measure of the dot-like-ness of each area
    Dot-like-ness is roughly how much this area differs from its most similar neighbor
    */
    private static float[][] dot_transform(float[][] img, float r)
    {
        int M = img.length;
        int N = img[0].length;
        float[][] output = new float[M][N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                output[i][j] = -Float.MAX_VALUE;
            }
        }
        
        for (float angle = 0; angle < 360; angle += 15)
        {
            int di1 = (int)(r*Math.sin(Math.toRadians(angle)));
            int dj1 = (int)(r*Math.cos(Math.toRadians(angle)));
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    int i1 = i + di1;
                    int j1 = j + dj1;
                    if (i1 < 0 || i1 >= M || j1 < 0 || j1 >= N)
                    {
                        continue;
                    }
                    output[i][j] = Math.max(output[i][j], img[i][j] - img[i1][j1]);
                }
            }

        }
        return output;
    }


    /*
    Transform image into a measure of the edge-like-ness of each area
    Edge-like-ness is roughly how much this area's least similar neighbors differ from each other, in a direction tangential to the direction to the area
    Because of the construction, areas immediately adjacent to an edge are still not considered edge-like; the edge has to pass through
    */
    private static float[][] spd_transform(float[][] img, float r, float angle_step)
    {
        int M = img.length;
        int N = img[0].length;
        float[][] output = new float[M][N];

        for (float angle = 0; angle < 360; angle += 15)
        {
            int di1 = (int)(r*Math.sin(Math.toRadians(angle)));
            int dj1 = (int)(r*Math.cos(Math.toRadians(angle)));
            int di2 = (int)(r*Math.sin(Math.toRadians(angle+angle_step)));
            int dj2 = (int)(r*Math.cos(Math.toRadians(angle+angle_step)));
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    int i1 = i + di1;
                    int j1 = j + dj1;
                    int i2 = i + di2;
                    int j2 = j + dj2;
                    if (i1 < 0 || i1 >= M || j1 < 0 || j1 >= N)
                    {
                        continue;
                    }
                    if (i2 < 0 || i2 >= M || j2 < 0 || j2 >= N)
                    {
                        continue;
                    }
                    output[i][j] = Math.max(output[i][j], Math.abs(img[i1][j1] - img[i2][j2]));
                }
            }
        }
        return output;
    }


    /*
    Calculate a scaling value for the other two transforms
    This makes them independent of the range of values in the image
    */
    private static float scale_value(float[][] img, float r)
    {
        int M = img.length;
        int N = img[0].length;
        int P = 8;
        // float[][][] output = new float[M][N][P];

        float sum = 0;
        float sumsq = 0;
        for (int k = 0; k < P; k++)
        {
            float angle = k*45;
            int di1 = (int)(r*Math.sin(Math.toRadians(angle)));
            int dj1 = (int)(r*Math.cos(Math.toRadians(angle)));
            for (int i = 0; i < M; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    int i1 = i + di1;
                    int j1 = j + dj1;
                    if (i1 < 0 || i1 >= M || j1 < 0 || j1 >= N)
                    {
                        continue;
                    }
                    // output[i][j][k] = img[i][j] - img[i1][j1];
                    float diff = img[i][j] - img[i1][j1];
                    sum += diff;
                    sumsq += diff * diff;
                }
            }
        }

        // return std_3d(output);
        return (float)Math.sqrt( sumsq/(M*N*P) - Math.pow( sum/(M*N*P), 2 ) ) + 0.001f;
    }


    /*
    Find microaneurysms
    Produces a mask that identifies the potential regions, with same dimensions as the source image
    */
    private static int[][] ma_detect(float[][] img, float[][] kernel, float s_threshold, float d_threshold, float s_r, float d_r, float angle_step)
    {
        int M = img.length;
        int N = img[0].length;
        float[][] img_filt = convolve_2d(img, kernel);

        float[][] d = dot_transform(img_filt, d_r);
        float[][] s = spd_transform(img_filt, s_r, angle_step);

        float scale = scale_value(img_filt, d_r);
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                d[i][j] = d[i][j] / scale;
                s[i][j] = s[i][j] / scale;
            }
        }

        int[][] ma_image = new int[M][N];
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if ( (s[i][j] < s_threshold) && (d[i][j] < d_threshold) )
                {
                    ma_image[i][j] = 1;
                }
            }
        }

        return ma_image;
    }


    /*
    Functions from scipy, ndimage and skimage which are missing in java
    */
    private static float std_3d(float[][][] data)
    {
        int M = data.length;
        int N = data[0].length;
        int P = data[0][0].length;

        float sum = 0;
        float sumsq = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                for (int k = 0; k < P; k++)
                {
                    sum += data[i][j][k];
                    sumsq += data[i][j][k]*data[i][j][k];
                }
            }
        }

        return (float)Math.sqrt( sumsq/(M*N*P) - Math.pow( sum/(M*N*P), 2 ) );
    }


    private static float std_2d(float[][] data)
    {
        int M = data.length;
        int N = data[0].length;

        float sum = 0;
        float sumsq = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                sum += data[i][j];
                sumsq += data[i][j]*data[i][j];
            }
        }

        return (float)Math.sqrt( sumsq/(M*N) - Math.pow( sum/(M*N), 2 ) );
    }


    private static float[][] convolve_2d(float[][] img, float[][] kernel)
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


    private static int[][] connected_components_2d(int[][] img)
    {
        int M = img.length;
        int N = img[0].length;

        int[][] labels = new int[M][N];
        int current_label = 0;

        int[] di = { -1,  0,  1,  -1,  1,  -1, 0, 1 };
        int[] dj = { -1, -1, -1,   0,  0,   1, 1, 1 };

        ArrayDeque<Point> queue = new ArrayDeque<Point>();

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (img[i][j] != 0 && labels[i][j] == 0) 
                {
                    current_label ++;
                    labels[i][j] = current_label;
                    queue.push(new Point(i,j));
                }

                while (! queue.isEmpty())
                {
                    Point p = queue.pop();
                    int i1 = p.x;
                    int j1 = p.y;
                    for (int direction = 0; direction < di.length; direction++)
                    {
                        int i2 = i1 + di[direction];
                        int j2 = j1 + dj[direction];
                        if (i2 < 0 || i2 >= M || j2 < 0 || j2 >= N)
                        {
                            continue;
                        }

                        if (img[i2][j2] != 0 && labels[i2][j2] == 0) 
                        {
                            labels[i2][j2] = current_label;
                            queue.push(new Point(i2,j2));
                        }
                    }
                }

            }
        }

        return labels;
    }


    /*
    Debugging utilities
    */
    private static void print_array_2d(int[][] img)
    {
        int M = img.length;
        int N = img[0].length;

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                System.out.print(String.format("%d,", img[i][j]));
            }
            System.out.print("\n");
        }
    }


    private static void imsave(float[][] img, String filename) throws IOException
    {
        int M = img.length;
        int N = img[0].length;
        BufferedImage bimg = new BufferedImage(M, N, BufferedImage.TYPE_INT_RGB);

        float s = std_2d(img);

        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int c = (int)( 255 * img[i][j] / (3*s) );
                if (c > 255) { c = 255; }
                if (c < 0) { c = 0; }
                bimg.setRGB(i, j, ((c << 16) | (c << 8) | c));
            }
        }

        ImageIO.write(bimg, "png", new File(filename));

        System.out.println(String.format("imsave %s scale=%f", filename, s));
    }


}
