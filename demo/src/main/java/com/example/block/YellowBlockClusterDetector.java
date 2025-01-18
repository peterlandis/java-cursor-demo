package com.example.block;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class YellowBlockClusterDetector {

    // Constants for yellow color range in HSV
    private static final Scalar LOWER_YELLOW = new Scalar(20, 100, 100);
    private static final Scalar UPPER_YELLOW = new Scalar(30, 255, 255);

    private static final double MIN_CLUSTER_AREA = 500.0;
    private static final double MIN_BLOCK_AREA = 100.0;

    /**
     * Detect yellow blocks in clusters from an image.
     *
     * @param imagePath  Path to the input image.
     * @param outputPath Path to save the annotated output image.
     */
    public void detectYellowBlocks(String imagePath, String outputPath) {
        Mat inputImage = Imgcodecs.imread(imagePath);
        if (inputImage.empty()) {
            throw new RuntimeException("Image could not be loaded.");
        }

        Mat hsvImage = new Mat();
        Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV);

        Mat yellowMask = new Mat();
        Core.inRange(hsvImage, LOWER_YELLOW, UPPER_YELLOW, yellowMask);

        // Improved morphological operations
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.erode(yellowMask, yellowMask, kernel);
        Imgproc.dilate(yellowMask, yellowMask, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(yellowMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            System.out.println("No yellow clusters detected.");
            return;
        }

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < MIN_CLUSTER_AREA) continue;

            Rect clusterRect = Imgproc.boundingRect(contour);
            System.out.println("Detected yellow cluster at: " + clusterRect);

            Mat clusterMask = Mat.zeros(yellowMask.size(), CvType.CV_8UC1);
            Imgproc.drawContours(clusterMask, List.of(contour), -1, new Scalar(255), -1);

            // Split clusters into blocks using refined criteria
            splitClusterIntoBlocks(clusterMask, inputImage, clusterRect);
        }

        Imgcodecs.imwrite(outputPath, inputImage);
        System.out.println("Annotated image saved at: " + outputPath);
    }

    /**
     * Split large clusters into individual blocks.
     *
     * @param clusterMask Cluster mask.
     * @param outputImage Output image.
     * @param clusterRect Cluster bounding box.
     */
    private void splitClusterIntoBlocks(Mat clusterMask, Mat outputImage, Rect clusterRect) {
        Mat subRegionMask = clusterMask.submat(clusterRect);

        List<MatOfPoint> blockContours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(subRegionMask, blockContours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint blockContour : blockContours) {
            double blockArea = Imgproc.contourArea(blockContour);
            if (blockArea < MIN_BLOCK_AREA) continue;

            Rect blockRect = Imgproc.boundingRect(blockContour);
            System.out.println("Detected block at: " + blockRect);

            // Adjust coordinates relative to the original image
            Rect absoluteBlockRect = new Rect(
                    blockRect.x + clusterRect.x,
                    blockRect.y + clusterRect.y,
                    blockRect.width,
                    blockRect.height
            );

            Imgproc.rectangle(outputImage, absoluteBlockRect, new Scalar(0, 255, 0), 2);
        }
    }
}
