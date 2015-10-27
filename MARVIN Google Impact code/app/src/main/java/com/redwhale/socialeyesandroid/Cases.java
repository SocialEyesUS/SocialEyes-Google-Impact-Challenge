package com.redwhale.socialeyesandroid;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.BaseAdapter;
import android.widget.GridView;
import android.widget.ImageView;
import android.widget.TextView;


public class Cases extends ActionBarActivity {

    class CaseAdapter extends BaseAdapter {
        private int[] cases;

        public CaseAdapter(int[] cases) {
            super();
            this.cases = cases;
        }

        @Override
        public int getCount() {
            return cases.length;
        }

        @Override
        public Object getItem(int position) {
            return cases[position];
        }

        @Override
        public long getItemId(int position) {
            return (long) cases[position];
        }

        @Override
        public View getView(final int position, View convertView, ViewGroup parent) {
            LayoutInflater inflater = (LayoutInflater) getApplicationContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            View rowView = inflater.inflate(R.layout.grid_case, null);
            ImageView imgView = (ImageView) rowView.findViewById(R.id.grid_case_img);
            Bitmap img = storage.getImg(cases[position], null);
            imgView.setImageBitmap(img);
            TextView txt = (TextView) rowView.findViewById(R.id.grid_case_txt);
            txt.setText("Case " + Integer.toString(cases[position]));

            rowView.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    Intent intent = new Intent(getApplicationContext(), Home.class);
                    intent.putExtra("case", cases[position]);
                    setResult(RESULT_OK, intent);
                    finish();
                }
            });

            return rowView;
        }
    }

    private Storage storage;
    private static int[] cases;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cases);

        storage = new Storage(getApplicationContext());
        cases = storage.getCases();

        GridView gridView = (GridView) findViewById(R.id.gridview);
        gridView.setAdapter(new CaseAdapter(cases));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_cases, menu);
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

    public void backButtonClicked(View view) {
        Intent intent = new Intent(this, Home.class);
        startActivity(intent);
    }
}


