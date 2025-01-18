package com.example.block;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import nu.pattern.OpenCV;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class YellowBlockClusterDetectorTest {

    private YellowBlockClusterDetector detector;
    private String testImagePath;

    @TempDir
    Path tempDir;

    @BeforeAll
    static void initOpenCV() {
        // Load OpenCV native library
        OpenCV.loadLocally();
    }

    @BeforeEach
    void setUp() {
        detector = new YellowBlockClusterDetector();
        // Assuming the test image is in src/test/resources
        testImagePath = getClass().getResource("/block_cluster.jpeg").getPath();
    }

    @Test
    void detectYellowBlocks_WithValidImage_ShouldProcessSuccessfully() {
        // Arrange
        String outputPath = testImagePath + "/output.jpg";

        System.out.println(outputPath);

        // Act
        assertDoesNotThrow(() -> {
            detector.detectYellowBlocks(testImagePath, outputPath);
        });

        // Assert
        File outputFile = new File(outputPath);
        assertThat(outputFile)
            .exists()
            .isFile()
            .canRead();
    }

    @Test
    void detectYellowBlocks_WithInvalidImagePath_ShouldThrowException() {
        // Arrange
        String invalidPath = "nonexistent.jpg";
        String outputPath = tempDir.resolve("output.jpg").toString();

        // Act & Assert
        assertThrows(RuntimeException.class, () -> {
            detector.detectYellowBlocks(invalidPath, outputPath);
        });
    }

    @Test
    void detectYellowBlocks_WithInvalidOutputPath_ShouldThrowException() {
        // Arrange
        String invalidOutputPath = "/Users/peterlandis/Development/java_projects/demo/src/test/resources/output.jpg";

        // Act & Assert
        assertThrows(RuntimeException.class, () -> {
            detector.detectYellowBlocks(testImagePath, invalidOutputPath);
        });
    }
}