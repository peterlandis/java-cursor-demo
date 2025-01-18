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
    private static final Scalar LOWER_YELLOW = new Scalar(20, 100, 100); // Adjust as needed
    private static final Scalar UPPER_YELLOW = new Scalar(30, 255, 255); // Adjust as needed

    // Minimum area threshold for filtering small contours
    private static final double MIN_CLUSTER_AREA = 500.0;
    private static final double MIN_BLOCK_AREA = 100.0;

    /**
     * Method to detect yellow blocks in clusters from an image.
     *
     * @param imagePath Path to the input image.
     * @param outputPath Path to save the annotated output image.
     */
    public void detectYellowBlocks(String imagePath, String outputPath) {
        // Load the input image
        Mat inputImage = Imgcodecs.imread(imagePath);
        if (inputImage.empty()) {
            throw new RuntimeException("Image could not be loaded.");
        }

        // Convert the image to HSV color space
        Mat hsvImage = new Mat();
        Imgproc.cvtColor(inputImage, hsvImage, Imgproc.COLOR_BGR2HSV);

        // Create a binary mask for yellow color
        Mat yellowMask = new Mat();
        Core.inRange(hsvImage, LOWER_YELLOW, UPPER_YELLOW, yellowMask);

        // Apply morphological operations to clean up noise
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(yellowMask, yellowMask, Imgproc.MORPH_CLOSE, kernel);

        // Find contours to detect clusters
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(yellowMask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.isEmpty()) {
            System.out.println("No yellow clusters detected.");
            return;
        }

        // Process each detected cluster
        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);
            if (area < MIN_CLUSTER_AREA) {
                continue; // Skip small areas (likely noise)
            }

            // Draw a bounding box around the detected cluster
            Rect clusterRect = Imgproc.boundingRect(contour);
            System.out.println("Detected yellow cluster at: " + clusterRect);

            // Create a mask for the cluster
            Mat clusterMask = Mat.zeros(yellowMask.size(), CvType.CV_8UC1);
            Imgproc.drawContours(clusterMask, List.of(contour), -1, new Scalar(255), -1);

            // Detect individual blocks within the cluster
            detectIndividualBlocks(clusterMask, inputImage);
        }

        // Save the output image with annotations
        Imgcodecs.imwrite(outputPath, inputImage);
        System.out.println("Annotated image saved at: " + outputPath);
    }

    /**
     * Method to detect individual blocks within a cluster mask.
     *
     * @param clusterMask Mask of the cluster.
     * @param outputImage Image on which to draw detected blocks.
     */
    private void detectIndividualBlocks(Mat clusterMask, Mat outputImage) {
        // Detect contours of individual blocks within the cluster
        List<MatOfPoint> blockContours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(clusterMask, blockContours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint blockContour : blockContours) {
            double blockArea = Imgproc.contourArea(blockContour);
            if (blockArea < MIN_BLOCK_AREA) {
                continue; // Skip very small contours
            }

            // Draw a bounding box around each detected block
            Rect blockRect = Imgproc.boundingRect(blockContour);
            System.out.println("Detected individual yellow block at: " + blockRect);

            // Draw the block rectangle on the output image
            Imgproc.rectangle(outputImage, blockRect, new Scalar(0, 255, 0), 2); // Green bounding box
        }
    }
}
