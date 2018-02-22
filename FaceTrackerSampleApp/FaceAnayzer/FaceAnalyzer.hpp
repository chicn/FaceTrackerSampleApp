#pragma once

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "Face.h"

#pragma GCC diagnostic ignored "-Wconditional-uninitialized"
#pragma GCC diagnostic ignored "-Wdocumentation"
#pragma GCC diagnostic ignored "-Wconversion"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#pragma GCC diagnostic pop

class FaceAnalyzer {
public:
    FaceAnalyzer();
    void run(cv::Mat image);
    void run(cv::Mat& image, std::vector<Face> faces);

private:

    dlib::frontal_face_detector faceDetector;
    dlib::shape_predictor landmarkPredictor;

    // Initialize Optical Flow
    bool firstLoop;
    cv::Mat prevgray, gray;
    std::vector<cv::Point2f> prevTrackPts;
    std::vector<cv::Point2f> nextTrackPts;
};
