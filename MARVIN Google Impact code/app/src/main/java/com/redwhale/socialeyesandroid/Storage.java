package com.redwhale.socialeyesandroid;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;


public class Storage {
    private File root;

    public Storage(Context c) {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)) {
            File external = Environment.getExternalStorageDirectory();
            root = new File(external, "SocialEyes");
            root.mkdir();
        }
        else {
            root = c.getFilesDir();
            root.mkdir();
        }
    }

    public File getRoot() {
        return root;
    }

    public void saveImg(int caseNum, String ext, Bitmap img) throws FileNotFoundException {
        File caseFolder = new File(root, Integer.toString(caseNum));
        caseFolder.mkdirs();
        String filename = Integer.toString(caseNum) + "_" + ext + ".jpg";
        File imgFile = new File(caseFolder, filename);
        if (imgFile == null)
            Log.i("kjl", "imgfile null");
        FileOutputStream out = new FileOutputStream(imgFile);
        if (out == null)
            Log.i("kjl", "out null");

        img.compress(Bitmap.CompressFormat.JPEG, 100, out);

        try {
            out.close();
        }
        catch (IOException e) {
            Log.e("kjl", e.getMessage());
        }
    }

    public Bitmap getImg(int caseNum, String ext) {
        File caseFolder = new File(root, Integer.toString(caseNum));
        String filename = Integer.toString(caseNum) + "_" + ext + ".jpg";
        File imgFile = new File(caseFolder, filename);
        return BitmapFactory.decodeFile(imgFile.getPath());
    }

    public int[] getCases() {
        String[] dirs = root.list(new FilenameFilter() {
            @Override
            public boolean accept(File dir, String filename) {
                return (new File(dir, filename)).isDirectory();
            }
        });
        int[] cases = new int[dirs.length];
        for (int i = 0; i < cases.length; ++i) {
            cases[i] = Integer.parseInt(dirs[i]);
        }
        return cases;
    }

    public Bitmap[] getCaseImages(int caseNum) {
        File caseFolder = new File(root, Integer.toString(caseNum));
        File[] caseFiles = caseFolder.listFiles();
        Bitmap[] imgs = new Bitmap[caseFiles.length];
        for (int i = 0; i < imgs.length; ++i) {
            imgs[i] = BitmapFactory.decodeFile(caseFiles[i].getPath());
        }
        return imgs;
    }
}
