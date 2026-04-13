package com.example.weblauncher;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.util.List;

public class AppAdapter extends RecyclerView.Adapter<AppAdapter.ViewHolder> {

    private final List<WebApp> apps;
    private final OnAppClickListener clickListener;
    private final OnAppLongClickListener longClickListener;

    public interface OnAppClickListener { void onClick(WebApp app); }
    public interface OnAppLongClickListener { void onLongClick(WebApp app, int position); }

    public AppAdapter(List<WebApp> apps, OnAppClickListener click, OnAppLongClickListener longClick) {
        this.apps = apps;
        this.clickListener = click;
        this.longClickListener = longClick;
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View v = LayoutInflater.from(parent.getContext()).inflate(R.layout.item_app, parent, false);
        return new ViewHolder(v);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        WebApp app = apps.get(position);
        holder.tvName.setText(app.name);
        holder.ivFavicon.setImageBitmap(makeLetterIcon(holder.itemView.getContext(), app.name));
        holder.itemView.setOnClickListener(v -> clickListener.onClick(app));
        holder.itemView.setOnLongClickListener(v -> {
            longClickListener.onLongClick(app, holder.getAdapterPosition());
            return true;
        });
    }

    @Override
    public int getItemCount() { return apps.size(); }

    // 生成字母图标
    private Bitmap makeLetterIcon(Context ctx, String name) {
        int size = (int) (ctx.getResources().getDisplayMetrics().density * 48);
        Bitmap bmp = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bmp);

        int[] colors = {0xFF5C6BC0, 0xFF26A69A, 0xFFEF5350, 0xFFAB47BC,
                        0xFF42A5F5, 0xFFFF7043, 0xFF66BB6A, 0xFF26C6DA};
        int colorIdx = Math.abs(name.hashCode()) % colors.length;

        Paint bg = new Paint(Paint.ANTI_ALIAS_FLAG);
        bg.setColor(colors[colorIdx]);
        canvas.drawCircle(size / 2f, size / 2f, size / 2f, bg);

        Paint text = new Paint(Paint.ANTI_ALIAS_FLAG);
        text.setColor(Color.WHITE);
        text.setTextSize(size * 0.4f);
        text.setTextAlign(Paint.Align.CENTER);

        String letter = name.isEmpty() ? "?" : name.substring(0, 1).toUpperCase();
        Rect bounds = new Rect();
        text.getTextBounds(letter, 0, letter.length(), bounds);
        canvas.drawText(letter, size / 2f, size / 2f + bounds.height() / 2f, text);

        return bmp;
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        ImageView ivFavicon;
        TextView tvName;
        ViewHolder(View v) {
            super(v);
            ivFavicon = v.findViewById(R.id.ivFavicon);
            tvName = v.findViewById(R.id.tvName);
        }
    }
}
