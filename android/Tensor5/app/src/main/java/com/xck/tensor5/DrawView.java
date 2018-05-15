package com.xck.tensor5;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

public class DrawView extends View {
    private Paint mPaint = new Paint();
    private DrawModel mModel;
    // 28x28 pixel Bitmap
    private Bitmap mOffscreenBitmap;
    private Canvas mOffscreenCanvas;
    private Matrix mMatrix = new Matrix();
    private Matrix mInvMatrix = new Matrix();
    private int mDrawnLineSize = 0;
    private boolean mSetuped = false;
    private float mTmpPoints[] = new float[2];
    private float mLastX;
    private float mLastY;
    private PointF mTmpPoint = new PointF();

    public DrawView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void setModel(DrawModel model) {
        this.mModel = model;
    }

    public void reset() {
        mDrawnLineSize = 0;
        if (mOffscreenBitmap != null) {
            mPaint.setColor(Color.WHITE);
            int width = mOffscreenBitmap.getWidth();
            int height = mOffscreenBitmap.getHeight();
            mOffscreenCanvas.drawRect(new Rect(0, 0, width, height), mPaint);
        }
    }

    private void setup() {
        mSetuped = true;
        // View size
        float width = getWidth();
        float height = getHeight();
        // Model (bitmap) size
        float modelWidth = mModel.getWidth();
        float modelHeight = mModel.getHeight();
        float scaleW = width / modelWidth;
        float scaleH = height / modelHeight;
        float scale = scaleW;
        if (scale > scaleH) {
            scale = scaleH;
        }

        float newCx = modelWidth * scale / 2;
        float newCy = modelHeight * scale / 2;
        float dx = width / 2 - newCx;
        float dy = height / 2 - newCy;

        mMatrix.setScale(scale, scale);
        mMatrix.postTranslate(dx, dy);
        mMatrix.invert(mInvMatrix);
        mSetuped = true;
    }

    @Override
    public void onDraw(Canvas canvas) {
        if (mModel == null) {
            return;
        }
        if (!mSetuped) {
            setup();
        }
        if (mOffscreenBitmap == null) {
            return;
        }
        int startIndex = mDrawnLineSize - 1;
        if (startIndex < 0) {
            startIndex = 0;
        }
        drawRenderer(mOffscreenCanvas, mModel, mPaint, startIndex);
        canvas.drawBitmap(mOffscreenBitmap, mMatrix, mPaint);
        mDrawnLineSize = mModel.getLineSize();
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        int action = event.getAction() & MotionEvent.ACTION_MASK;
        if (action == MotionEvent.ACTION_DOWN) {
            processTouchDown(event);
            return true;
        } else if (action == MotionEvent.ACTION_MOVE) {
            processTouchMove(event);
            return true;
        } else if (action == MotionEvent.ACTION_UP) {
            processTouchUp();
            return true;
        }
        return false;
    }

    private void drawRenderer(Canvas canvas, DrawModel model, Paint paint,
                              int startLineIndex) {
        paint.setAntiAlias(true);
        int lineSize = model.getLineSize();
        for (int i = startLineIndex; i < lineSize; ++i) {
            DrawModel.Line line = model.getLine(i);
            paint.setColor(Color.BLACK);
            int elemSize = line.getElemSize();
            if (elemSize < 1) {
                continue;
            }
            DrawModel.LineElem elem = line.getElem(0);
            float lastX = elem.x;
            float lastY = elem.y;

            for (int j = 0; j < elemSize; ++j) {
                elem = line.getElem(j);
                float x = elem.x;
                float y = elem.y;
                canvas.drawLine(lastX, lastY, x, y, paint);
                lastX = x;
                lastY = y;
            }
        }
    }

    /**
     * Convert screen position to local pos (pos in bitmap)
     */
    public void calcPos(float x, float y, PointF out) {
        mTmpPoints[0] = x;
        mTmpPoints[1] = y;
        mInvMatrix.mapPoints(mTmpPoints);
        out.x = mTmpPoints[0];
        out.y = mTmpPoints[1];
    }

    public void onResume() {
        createBitmap();
    }

    public void onPause() {
        releaseBitmap();
    }

    private void createBitmap() {
        if (mOffscreenBitmap != null) {
            mOffscreenBitmap.recycle();
        }
        mOffscreenBitmap = Bitmap.createBitmap(mModel.getWidth(), mModel.getHeight(), Bitmap.Config.ARGB_8888);
        mOffscreenCanvas = new Canvas(mOffscreenBitmap);
        reset();
    }

    private void releaseBitmap() {
        if (mOffscreenBitmap != null) {
            mOffscreenBitmap.recycle();
            mOffscreenBitmap = null;
            mOffscreenCanvas = null;
        }
        reset();
    }

    /**
     * Get 28x28 pixel data for tensorflow input.
     */
    public float[] getPixelData() {
        if (mOffscreenBitmap == null) {
            return null;
        }

        int width = mOffscreenBitmap.getWidth();
        int height = mOffscreenBitmap.getHeight();

        // Get 28x28 pixel data from bitmap
        int[] pixels = new int[width * height];
        mOffscreenBitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        float[] retPixels = new float[pixels.length];
        for (int i = 0; i < pixels.length; ++i) {
            // Set 0 for white and 255 for black pixel
            int pix = pixels[i];
            int b = pix & 0xff;
            retPixels[i] = 0xff - b;
        }
        return retPixels;
    }

    private void processTouchDown(MotionEvent event) {
        mLastX = event.getX();
        mLastY = event.getY();
        calcPos(mLastX, mLastY, mTmpPoint);
        float lastConvX = mTmpPoint.x;
        float lastConvY = mTmpPoint.y;
        mModel.startLine(lastConvX, lastConvY);
    }

    private void processTouchMove(MotionEvent event) {
        float x = event.getX();
        float y = event.getY();

        calcPos(x, y, mTmpPoint);
        float newConvX = mTmpPoint.x;
        float newConvY = mTmpPoint.y;
        mModel.addLineElem(newConvX, newConvY);
        mLastX = x;
        mLastY = y;
        invalidate();
    }

    private void processTouchUp() {
        mModel.endLine();
    }
}

