#pragma once

#include <array>
#include <opencv2/core.hpp>

constexpr int LANDMARK_COUNT = 68;
using LandmarkArray = std::array<cv::Point, LANDMARK_COUNT>;
struct Face {
    int faceID;
    cv::Rect rect;
    LandmarkArray landmarks;
};
