#pragma once

#include <opencv2/core.hpp>
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
    void run(const cv::Mat& image, std::vector<Face> faces);

private:
    constexpr static float SCALE = 0.5;

    dlib::frontal_face_detector faceDetector;
    dlib::shape_predictor landmarkPredictor;
};
