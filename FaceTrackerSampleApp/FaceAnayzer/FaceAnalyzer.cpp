#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include "FaceAnalyzer.hpp"
#include "ResourcePath.hpp"

namespace {
    static const float SCALE = 0.5;
}

FaceAnalyzer::FaceAnalyzer() {
    faceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(resourcePath("shape_predictor_68_face_landmarks", "dat")) >> landmarkPredictor;

    firstLoop = true;

    std::vector<cv::Point2f> initialVal(68, cv::Point2f(0.0, 0.0));
    prevTrackPts = initialVal;
    nextTrackPts = initialVal;
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

void FaceAnalyzer::run(cv::Mat image) {

    // Resize
    cv::Mat small, smallGray;
    cv::resize(image, small, cv::Size(), SCALE, SCALE, cv::INTER_NEAREST);
    cv::cvtColor(small, smallGray, CV_BGR2GRAY);

    dlib::cv_image<uint8_t> cimg(smallGray);

    std::vector<dlib::rectangle> faces = faceDetector(cimg);

    std::vector<dlib::full_object_detection> shapes;
    for (unsigned long i = 0; i < faces.size(); ++i)
        shapes.push_back(landmarkPredictor(cimg, faces[i]));

    if (firstLoop) {
        if (!shapes.empty()) {
            prevgray = smallGray.clone();
            const dlib::full_object_detection& d = shapes[0];
            for (int i = 0; i < d.num_parts(); i++) {
                prevTrackPts[i].x = d.part(i).x();
                prevTrackPts[i].y = d.part(i).y();
            }
            firstLoop = false;
        } else {
            return;
        }
    }

    if (shapes.size() == 1) {
        gray = smallGray.clone();
        if (prevgray.data) {
            std::vector<uchar> status;
            std::vector<float> err;
            calcOpticalFlowPyrLK(prevgray, gray, prevTrackPts, nextTrackPts, status, err);
            double diff = calDistanceDiff(prevTrackPts, nextTrackPts);
            std::cout << "variance:" << diff << std::endl;
            if (diff > 0.1) {
                const dlib::full_object_detection& d = shapes[0];
                for (int i = 0; i < d.num_parts(); i++) {
                    cv::circle(image, cv::Point2f(d.part(i).x(), d.part(i).y())/SCALE, 2, cv::Scalar(0, 255, 255), -1);
                    nextTrackPts[i].x = d.part(i).x();
                    nextTrackPts[i].y = d.part(i).y();
                }
            } else {
                for (int i = 0; i < nextTrackPts.size(); i++) {
                    cv::circle(image, nextTrackPts[i]/SCALE, 2, cv::Scalar(0, 0, 255), 1);
                }
            }
            std::swap(prevTrackPts, nextTrackPts);
            std::swap(prevgray, gray);
        }
    }
}
