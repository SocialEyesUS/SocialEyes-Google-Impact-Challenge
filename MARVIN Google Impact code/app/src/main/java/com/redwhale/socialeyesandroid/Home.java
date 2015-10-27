package com.redwhale.socialeyesandroid;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Typeface;
import android.graphics.drawable.BitmapDrawable;
import android.os.AsyncTask;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.content.Intent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;

import uk.co.senab.photoview.PhotoViewAttacher;


public class Home extends ActionBarActivity {
    private ImageView image;
    private PhotoViewAttacher attacher;

    private int caseID = 0;

    private Storage storage;

    private int activeFilter =  0; // represents the id of the active button
    private int toneStatus = 0;
    private int sharpenStatus = 0;

    private class Transform extends AsyncTask<String, Void, Bitmap> {
        private Button button;
        private boolean isPreexisting = true;

        private Transform(int buttonID) {
            this.button = (Button) findViewById(buttonID);
        }

        @Override
        protected void onPreExecute() {
            button.setEnabled(false);
            button.setBackgroundResource(R.drawable.button_processing);
        }

        @Override
        protected Bitmap doInBackground(String... params) {
            Bitmap img = storage.getImg(caseID, null);
            Bitmap newImg = storage.getImg(caseID, params[0]);

            switch (params[0]) {
                case "color":
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.color(img);
                    }
                    break;
                case "rf":
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.redfree(img);
                    }
                    break;
                case "tone0":
                    newImg = storage.getImg(caseID, params[0]);
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.tone(img, 0.8f);
                    }
                    break;
                case "tone1":
                    newImg = storage.getImg(caseID, params[0]);
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.tone(img, 1.2f);
                    }
                    break;
                case "sharpen0":
                    newImg = storage.getImg(caseID, params[0]);
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.sharpen(img, 1.0f);
                    }
                    break;
                case "sharpen1":
                    newImg = storage.getImg(caseID, params[0]);
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.sharpen(img, 3.0f);
                    }
                    break;
                case "norm":
                    if (newImg == null) {
                        isPreexisting = false;
                        newImg = ImageProc.normalize(img);
                    }
                    break;
                default:
                    break;
            }

            if (!isPreexisting) {
                try {
                    switch (params[0]) {
                        case "tone":
                            storage.saveImg(caseID, params[0] + Integer.toString(toneStatus), newImg);
                            toneStatus++;
                            break;
                        case "sharpen":
                            storage.saveImg(caseID, params[0] + Integer.toString(sharpenStatus), newImg);
                            sharpenStatus++;
                            break;
                        default:
                            storage.saveImg(caseID, params[0], newImg);
                            break;
                    }
                } catch (FileNotFoundException e) {
                    Log.e("Error", "cannot save image to file");
                }
            }

            return newImg;
        }

        @Override
        protected void onPostExecute(Bitmap result) {
            button.setBackgroundResource(R.drawable.button_active);

            image.setImageDrawable(new BitmapDrawable(getResources(), result));
            attacher.update();

            button.setEnabled(true);
        }
    }

    private void preloadImages() {
        File pre = new File(storage.getRoot(), "26");
        if (pre.exists())
            return;

        Bitmap img1 = BitmapFactory.decodeResource(getResources(), R.drawable.eye_26);
        try {
            storage.saveImg(26, null, img1);
        }
        catch (FileNotFoundException e) {
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_home);

        image = (ImageView) findViewById(R.id.image);
        attacher = new PhotoViewAttacher(image);
        storage = new Storage(getApplicationContext());
        preloadImages();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_home, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    private void resetFilter(int buttonID) {
        Button button = (Button) findViewById(buttonID);
        if (button != null) {
            button.setTextSize(14);
            button.setBackgroundResource(R.drawable.button_default);
        }

        switch (activeFilter) {
            case R.id.rf_button:
                break;
            case R.id.tone_button:
                toneStatus = 0;
                button.setText("Tone");
                break;
            case R.id.sharpen_button:
                sharpenStatus = 0;
                button.setText("Sharpen");
                break;
            case R.id.norm_button:
                break;
            default:
                break;
        }

        activeFilter = 0;
    }

    private void loadOriginal() {
        Bitmap img = storage.getImg(caseID, null);
        if (img != null) {
            image.setImageBitmap(img);
            attacher.update();
            image.postInvalidate();
        }
    }

    private boolean isImageNull() {
        if (caseID == 0) {
            String message = "First select an image from the cases gallery";
            Toast toast = Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG);
            toast.show();
            return true;
        } else {
            return false;
        }
    }

    public void colorButtonClicked(View view) {
        int colorID = R.id.color_button;

        if (isImageNull()) {
            return;
        }

        if (activeFilter == colorID) {
            loadOriginal();
            resetFilter(colorID);
            return;
        }

        resetFilter(activeFilter);
        activeFilter = colorID;

        Transform colorTask = new Transform(colorID);
        colorTask.execute("color");
    }

    public void rfButtonClicked(View view) {
        int rfID = R.id.rf_button;

        if (isImageNull()) {
            return;
        }

        if (activeFilter == rfID) {
            loadOriginal();
            resetFilter(rfID);
            return;
        }

        resetFilter(activeFilter);
        activeFilter = rfID;

        Transform rfTask = new Transform(rfID);
        rfTask.execute("rf");
    }

    public void toneButtonClicked(View view) {
        int toneID = R.id.tone_button;

        if (isImageNull()) {
            return;
        }

        if (activeFilter != toneID) {
            resetFilter(activeFilter);
            activeFilter = toneID;
        }

        Button button = (Button) findViewById(R.id.tone_button);
        Transform toneTask;
        switch (toneStatus) {
            case 0:
                toneTask = new Transform(toneID);
                toneTask.execute("tone0");
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.NORMAL);
                button.setText("+");
                button.setTextSize(30);
                toneStatus++;
                break;
            case 1:
                toneTask = new Transform(toneID);
                toneTask.execute("tone1");
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.NORMAL);
                button.setText("-");
                button.setTextSize(30);
                toneStatus++;
                break;
            case 2:
                loadOriginal();
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.ITALIC);
                resetFilter(toneID);
                break;
            default:
                break;
        }
    }

    public void sharpenButtonClicked(View view) {
        int sharpenID = R.id.sharpen_button;

        if (isImageNull()) {
            return;
        }

        if (activeFilter != sharpenID) {
            resetFilter(activeFilter);
            activeFilter = sharpenID;
        }

        Button button = (Button) findViewById(R.id.sharpen_button);
        Transform sharpenTask;
        switch (sharpenStatus) {
            case 0:
                sharpenTask = new Transform(sharpenID);
                sharpenTask.execute("sharpen0");
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.NORMAL);
                button.setText("+");
                button.setTextSize(30);
                sharpenStatus++;
                break;
            case 1:
                sharpenTask = new Transform(sharpenID);
                sharpenTask.execute("sharpen1");
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.NORMAL);
                button.setText("++");
                button.setTextSize(30);
                sharpenStatus++;
                break;
            case 2:
                loadOriginal();
                button.setTypeface(Typeface.create(button.getTypeface(), Typeface.NORMAL), Typeface.ITALIC);
                resetFilter(sharpenID);
                break;
            default:
                break;
        }
    }

    public void normButtonClicked(View view) {
        int normId = R.id.norm_button;

        if (isImageNull()) {
            return;
        }

        if (activeFilter == normId) {
            loadOriginal();
            resetFilter(normId);
            return;
        }

        resetFilter(activeFilter);
        activeFilter = normId;

        Transform normTask = new Transform(normId);
        normTask.execute("norm");
    }

    public void casesButtonClicked(View view) {
        Intent intent = new Intent(this, Cases.class);
        startActivityForResult(intent, 1);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == 1) {
            if(resultCode == Activity.RESULT_OK){
                resetFilter(R.id.color_button);
                resetFilter(R.id.rf_button);
                resetFilter(R.id.tone_button);
                resetFilter(R.id.sharpen_button);
                resetFilter(R.id.norm_button);

                caseID = data.getIntExtra("case", 0);
                if (caseID != 0) {
                    Bitmap img = storage.getImg(caseID, null);
                    if (img != null) {
                        image.setImageBitmap(img);
                        attacher.update();
                    }
                    else {
                        String message = "Error: image cannot be loaded. Check filename for consistency";
                        Toast toast = Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG);
                        toast.show();
                    }
                }
            }
            if (resultCode == Activity.RESULT_CANCELED) {
                String message = "Error: Could not select a case to process";
                Toast toast = Toast.makeText(getApplicationContext(), message, Toast.LENGTH_LONG);
                toast.show();
            }
        }
    }
}
