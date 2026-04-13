package com.example.weblauncher;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.util.ArrayList;
import java.util.List;

public class LauncherActivity extends AppCompatActivity {

    private List<WebApp> apps = new ArrayList<>();
    private AppAdapter adapter;
    private SharedPreferences prefs;
    private static final String PREF_KEY = "apps";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // 全屏沉浸
        getWindow().getDecorView().setSystemUiVisibility(
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
            View.SYSTEM_UI_FLAG_FULLSCREEN |
            View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
            View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
        );
        setContentView(R.layout.activity_launcher);

        prefs = getSharedPreferences("weblauncher", MODE_PRIVATE);
        loadApps();

        RecyclerView rv = findViewById(R.id.recyclerView);
        rv.setLayoutManager(new GridLayoutManager(this, 5));
        adapter = new AppAdapter(apps, this::openApp, this::showDeleteDialog);
        rv.setAdapter(adapter);

        updateEmptyView();
        findViewById(R.id.btnAdd).setOnClickListener(v -> showAddDialog());
    }

    private void openApp(WebApp app) {
        Intent intent = new Intent(this, WebActivity.class);
        intent.putExtra("url", app.url);
        intent.putExtra("name", app.name);
        startActivity(intent);
    }

    private void showAddDialog() {
        View view = getLayoutInflater().inflate(R.layout.dialog_add, null);
        EditText etName = view.findViewById(R.id.etName);
        EditText etUrl = view.findViewById(R.id.etUrl);

        new AlertDialog.Builder(this)
            .setView(view)
            .setPositiveButton("添加", (d, w) -> {
                String name = etName.getText().toString().trim();
                String url = etUrl.getText().toString().trim();
                if (TextUtils.isEmpty(name) || TextUtils.isEmpty(url)) return;
                if (!url.startsWith("http")) url = "https://" + url;
                apps.add(new WebApp(name, url));
                adapter.notifyItemInserted(apps.size() - 1);
                saveApps();
                updateEmptyView();
            })
            .setNegativeButton("取消", null)
            .show();
    }

    private void showDeleteDialog(WebApp app, int position) {
        new AlertDialog.Builder(this)
            .setTitle(app.name)
            .setMessage("要删除这个网页吗？")
            .setPositiveButton("删除", (d, w) -> {
                apps.remove(position);
                adapter.notifyItemRemoved(position);
                saveApps();
                updateEmptyView();
            })
            .setNegativeButton("取消", null)
            .show();
    }

    private void updateEmptyView() {
        LinearLayout emptyView = findViewById(R.id.emptyView);
        emptyView.setVisibility(apps.isEmpty() ? View.VISIBLE : View.GONE);
    }

    private void saveApps() {
        prefs.edit().putString(PREF_KEY, new Gson().toJson(apps)).apply();
    }

    private void loadApps() {
        String json = prefs.getString(PREF_KEY, null);
        if (json != null) {
            apps = new Gson().fromJson(json, new TypeToken<List<WebApp>>(){}.getType());
        }
        if (apps == null) apps = new ArrayList<>();
    }

    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            getWindow().getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            );
        }
    }
}
