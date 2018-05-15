package com.xck.tensor5;

import android.graphics.RectF;

import java.util.List;

public interface BaseClassifer {

    /**
     * n immutable result returned by a Classifier describing what was recognized.
     */
    class Recognition{
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;

        /**
         * display name for recognition
         */
        private final String title;

        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;

        /**
         * Optional location within the source image for the location of the recognized object.
         * RextF是用来创建一个矩形的
         * RectF（float left,float top,float right,float bottom）构造一个指定了4个参数的矩形
         */
        private RectF location;

        /**
         * 构造方法
         */
        public Recognition(final String id, final String title, final Float confidence, final RectF location){
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;

        }

        public String getId() {
            return id;
        }

        public String getTitle() {
            return title;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return location;
        }

        public void setLocation(RectF location) {
            this.location = location;
        }


        @Override
        public String toString() {
            return "Recognition{" +
                    "id='" + id + '\'' +
                    ", title='" + title + '\'' +
                    ", confidence=" + confidence +
                    ", location=" + location +
                    '}';
        }
    }

    List<Recognition> recognizeImage (float[] pixels);

    void enableStatLogging(final boolean debug);
    String getStatString();

    void close();

}
