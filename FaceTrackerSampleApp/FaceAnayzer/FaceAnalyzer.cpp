#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include "FaceAnalyzer.hpp"
#include "ResourcePath.hpp"

FaceAnalyzer::FaceAnalyzer() {
    faceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(resourcePath("shape_predictor_68_face_landmarks", "dat")) >> landmarkPredictor;
}

namespace {

    double calDistanceDiff(std::vector<cv::Point2f> curPoints, std::vector<cv::Point2f> lastPoints) {
        double variance = 0.0;
        double sum = 0.0;
        std::vector<double> diffs;
        if (curPoints.size() == lastPoints.size()) {
            for (int i = 0; i < curPoints.size(); i++) {
                double diff = std::sqrt(std::pow(curPoints[i].x - lastPoints[i].x, 2.0) + std::pow(curPoints[i].y - lastPoints[i].y, 2.0));
                sum += diff;
                diffs.push_back(diff);
            }
            double mean = sum / diffs.size();
            for (int i = 0; i < curPoints.size(); i++) {
                variance += std::pow(diffs[i] - mean, 2);
            }
            return variance / diffs.size();
        }
        return variance;
    }
}

void FaceAnalyzer::run(cv::Mat& image, std::vector<Face> faces) {
    faces.clear();

    // resize image
    if (image.empty()) return;
    cv::Mat smImage, smGrImage;
    cv::resize(image, smImage, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
    cv::cvtColor(smImage, smGrImage, CV_BGR2GRAY);
    dlib::cv_image<uint8_t> dImage = dlib::cv_image<uint8_t>(smGrImage);

    // detect face
    std::vector<dlib::rectangle> rects = faceDetector(dImage);

    // detect face landmarks
    for (const auto& rect : rects) {
        dlib::full_object_detection shape = landmarkPredictor(dImage, rect);

        Face face;
        face.rect = cv::Rect(rect.left() / SCALE, rect.top() / SCALE, rect.width() / SCALE, rect.height() / SCALE);
        for (int i = 0; i < shape.num_parts(); ++i) {
            face.landmarks[i] = cv::Point(shape.part(i).x() / SCALE, shape.part(i).y() / SCALE);
        }
        faces.push_back(face);
    }

    for (const auto& face : faces) {
        cv::rectangle(image, face.rect, cv::Scalar(255));
        for (int i = 0; i < face.landmarks.size(); ++i)
            cv::circle(image, face.landmarks[i], 1, cv::Scalar(255,0,0), -1);
    }

    std::cout << "face count " << faces.size() << std::endl;
}
