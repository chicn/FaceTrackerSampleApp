#include "FaceAnalyzer.hpp"
#include "ResourcePath.hpp"

FaceAnalyzer::FaceAnalyzer() {
    faceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(resourcePath("shape_predictor_68_face_landmarks", "dat")) >> landmarkPredictor;
}

void FaceAnalyzer::run(const cv::Mat& image, std::vector<Face> faces) {

    // detect face
    
}
