package cn.alexchao.multicnn;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initialize();
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()) {
            case R.id.to_mnist:
                break;
            case R.id.to_cifar:
                gotoCifar();
                break;
            default:
                Toast.makeText(this, "no match", Toast.LENGTH_SHORT).show();
        }
    }

    private void initialize() {
        Button mnistBtn = (Button) findViewById(R.id.to_mnist);
        Button cifarBtn = (Button) findViewById(R.id.to_cifar);
        mnistBtn.setOnClickListener(this);
        cifarBtn.setOnClickListener(this);
    }

    private void gotoCifar() {
        Intent intent = new Intent();
        intent.setClass(MainActivity.this, CifarActivity.class);
        startActivity(intent);
    }
}
