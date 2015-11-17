// For desktop
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;

// For Android
// import android.graphics.Bitmap;

import java.util.Arrays;
import java.io.IOException;
import java.io.File;


public class ImageUtils
{

    // For desktop
    // NOTE this can be sped up a lot using bimg.getRaster().getPixels(), see http://stackoverflow.com/questions/6524196/java-get-pixel-array-from-image
    public static float[][][] buffered_image_to_array(BufferedImage bimg)
    {
        int channels = 3; // exactly 3 channels is required
        int rows = bimg.getHeight();
        int cols = bimg.getWidth();
        float[][][] img = new float[channels][rows][cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int p = bimg.getRGB(j, i);
                img[0][i][j] = ((p >> 16) & 0xFF) / 255.0f;
                img[1][i][j] = ((p >>  8) & 0xFF) / 255.0f;
                img[2][i][j] = ((p >>  0) & 0xFF) / 255.0f;
            }
        }

        return img;
    }

    public static float[][] buffered_image_to_array_green(BufferedImage bimg)
    {
        int rows = bimg.getHeight();
        int cols = bimg.getWidth();
        float[][] img = new float[rows][cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int p = bimg.getRGB(j, i);

                // pull out green from value
                img[i][j] = ((p >> 8) & 0xFF) / 255.0f;
            }
        }

        return img;
    }

    // NOTE this can be sped up by a lot using bimg.getRaster().setPixels(...), see http://stackoverflow.com/questions/14416107/int-array-to-bufferedimage
    public static BufferedImage array_to_buffered_image(float[][][] img)
    {
        int channels = img.length;
        int rows = img[0].length;
        int cols = img[0][0].length;
        BufferedImage bimg = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float r = img[0][i][j];
                float g = img[1][i][j];
                float b = img[2][i][j];

                int ri = Math.max(Math.min( (int)Math.round(255*r), 255), 0);
                int gi = Math.max(Math.min( (int)Math.round(255*g), 255), 0);
                int bi = Math.max(Math.min( (int)Math.round(255*b), 255), 0);

                int p = (ri << 16) | (gi << 8) | (bi << 0);
                bimg.setRGB( j, i, p );
            }
        }

        return bimg;
    }

    public static BufferedImage grayscale_array_to_buffered_image(float[][] img)
    {
        int rows = img.length;
        int cols = img[0].length;
        // maybe use TYPE_BYTE_GRAY
        BufferedImage bimg = new BufferedImage(cols, rows, BufferedImage.TYPE_INT_RGB);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float gray = img[i][j];

                int gray_i = Math.max(Math.min( (int)Math.round(255*gray), 255), 0);

                // TODO there is probably a better conversion here, but this is close
                int p = (gray_i << 16) | (gray_i << 8) | (gray_i << 0);
                bimg.setRGB( j, i, p );
            }
        }

        return bimg;
    }

    // For Android
    /*
    // NOTE this can be sped up with Bitmap.getPixels()
    public static float[][][] bitmap_to_array(Bitmap bimg)
    {
        int rows = bimg.getHeight();
        int cols = bimg.getWidth();
        int channels = 3; // exactly 3 channels is required
        float[][][] img = new float[rows][cols][channels];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                int p = bimg.getPixel(j, i);
                img[i][j][0] = ((p >> 16) & 0xFF) / 255.0f;
                img[i][j][1] = ((p >>  8) & 0xFF) / 255.0f;
                img[i][j][2] = ((p >>  0) & 0xFF) / 255.0f;
            }
        }

        return img;
    }

    // NOTE this can be sped up with Bitmap.createBitmap (int[] colors, ...)
    public static Bitmap array_to_bitmap(float[][][] img)
    {
        int rows = img.length;
        int cols = img[0].length;
        Bitmap bimg = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
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

    public static Bitmap grayscale_array_to_bitmap(float[][] img)
    {
        int rows = img.length;
        int cols = img[0].length;
        Bitmap bimg = Bitmap.createBitmap(cols, rows, Bitmap.Config.ARGB_8888);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                float gray = img[i][j];

                int gray_i = Math.max(Math.min( (int)Math.round(255*gray), 255), 0);

                // TODO there is probably a better conversion here, but this is close
                int p = (0xFF << 24) | (gray_i << 16) | (gray_i << 8) | (gray_i << 0);

                bimg.setPixel( j, i, p );
            }
        }

        return bimg;
    }
    */

}