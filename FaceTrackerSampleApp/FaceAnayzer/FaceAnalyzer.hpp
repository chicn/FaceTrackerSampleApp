#pragma once

#include <opencv2/core.hpp>
#include "Face.h"

#pragma GCC diagnostic ignored "-Wconditional-uninitialized"
#pragma GCC diagnostic ignored "-Wdocumentation"
#pragma GCC diagnostic ignored "-Wconversion"
#include <dlib/image_processing/shape_predictor.h>
#pragma GCC diagnostic pop

class FaceAnalyzer {
public:
    void run(const cv::Mat& image, std::vector<Face> faces);

private:
    dlib::shape_predictor shapePredictor;
};
