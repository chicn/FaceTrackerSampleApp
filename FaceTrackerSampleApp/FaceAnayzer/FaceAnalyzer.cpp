#include <opencv2/imgproc.hpp>
#include <dlib/opencv.h>
#include "FaceAnalyzer.hpp"
#include "ResourcePath.hpp"

namespace {
    static const int STATE_NUM = LANDMARK_COUNT * 4;
    static const int MEASURE_NUM = LANDMARK_COUNT * 2;
    static const float SCALE = 0.5;
}

FaceAnalyzer::FaceAnalyzer() {
    faceDetector = dlib::get_frontal_face_detector();
    dlib::deserialize(resourcePath("shape_predictor_68_face_landmarks", "dat")) >> landmarkPredictor;


    firstLoop = true;
    count = 0;
    redetected = true;

    std::vector<cv::Point2f> initialVal(68, cv::Point2f(0.0, 0.0));
    // for KF
    last_object = initialVal;
    kalman_points = initialVal;
    predict_points = initialVal;
    // for OF
    prevTrackPts = initialVal;
    nextTrackPts = initialVal;

    state = cv::Mat(STATE_NUM, 1, CV_32FC1);
    processNoise = cv::Mat(STATE_NUM, 1, CV_32FC1);
    measurement = cv::Mat::zeros(MEASURE_NUM, 1, CV_32F);

    // kf = std::make_unique<cv::KalmanFilter>(STATE_NUM, MEASURE_NUM);
    kf = std::unique_ptr<cv::KalmanFilter>(new cv::KalmanFilter(STATE_NUM, MEASURE_NUM));
    cv::randn(state, cv::Scalar::all(0), cv::Scalar::all(0.0));
    kf->transitionMatrix = cv::Mat::eye(STATE_NUM, STATE_NUM, CV_32FC1);

    //!< measurement matrix (H)
    setIdentity(kf->measurementMatrix);

    //!< process noise covariance matrix (Q)
    setIdentity(kf->processNoiseCov, cv::Scalar::all(1e-5));

    //!< measurement noise covariance matrix (R)
    setIdentity(kf->measurementNoiseCov, cv::Scalar::all(1e-1));

    //!< priori error estimate covariance matrix (P'(k)): P'(k)=A*P(k-1)*At + Q)*/  A代表F: transitionMatrix
    setIdentity(kf->errorCovPost, cv::Scalar::all(1));

    randn(kf->statePost, cv::Scalar::all(0), cv::Scalar::all(0.0));

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

//    if (firstLoop) {
//        prevgray = smGrImage.clone();
//        const LandmarkArray& pts = faces[0].landmarks;
//        for (int i = 0; i < pts.size(); ++i)
//            prevTrackPts[i] = pts[i];
//        firstLoop = false;
//    }

//    if (faces.size() == 1) {
//        const LandmarkArray& pts = faces[0].landmarks;
//        for (int i = 0; i < pts.size(); ++i)
//            kalman_points[i] = pts[i];
//    }

    if (faces.size() == 1) {
        if (firstLoop) {
            prevgray = smGrImage.clone();
            const LandmarkArray& pts = faces[0].landmarks;
            for (int i = 0; i < pts.size(); ++i)
                prevTrackPts[i] = pts[i];
            firstLoop = false;
        }


        const LandmarkArray& pts = faces[0].landmarks;
        for (int i = 0; i < pts.size(); ++i)
            kalman_points[i] = pts[i];

        cv::Mat predictionResult = kf->predict();
        for (int i = 0; i < LANDMARK_COUNT; ++i)
            predict_points[i] = cv::Point2f(predictionResult.at<float>(i * 2), predictionResult.at<float>(i * 2 + 1));

        gray = smGrImage.clone();
        if (prevgray.data) {
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prevgray, gray, prevTrackPts, nextTrackPts, status, err);
            double diff = calDistanceDiff(prevTrackPts, nextTrackPts);
            std::cout << "variance:" << diff << std::endl;

            if (diff > 1.0) {
                std::cout << "DLIB" << std::endl;
                const LandmarkArray& pts = faces[0].landmarks;
                for (int i = 0; i < pts.size(); ++i) {
                    cv::circle(image, pts[i], 2, cv::Scalar(0, 0, 255), -1);
                    nextTrackPts[i] = pts[i];
                }
            } else if (diff <= 1.0 && diff > 0.005) {
                std::cout<< "Optical Flow" << std::endl;
                for (int i = 0; i < nextTrackPts.size(); i++)
                    cv::circle(image, nextTrackPts[i], 2, cv::Scalar(255, 0, 0), -1);
            } else {
                std::cout<< "Kalman Filter" << std::endl;
                for (int i = 0; i < predict_points.size(); ++i) {
                    cv::circle(image, predict_points[i], 2, cv::Scalar(0, 255, 0), -1);
                    nextTrackPts[i] = predict_points[i];
                }
                redetected = false;
            }
        } else {
            redetected = true;
        }
        std::swap(prevTrackPts, nextTrackPts);
        std::swap(prevgray, gray);
    } else {
        redetected = true;
    }

    // draw?
    for (const auto& face : faces) {
        cv::rectangle(image, face.rect, cv::Scalar(255));
//        for (int i = 0; i < face.landmarks.size(); ++i)
            //cv::circle(image, face.landmarks[i], 1, cv::Scalar(255,0,0), -1);
            //cv::circle(image, nextTrackPts[i], 1, cv::Scalar(255,0,0), -1);
    }

    // Update Measurement
    for (int i = 0; i < measurement.rows; i++) {
        if (i % 2 == 0)
            measurement.at<float>(i) = (float)kalman_points[i / 2].x;
        else
            measurement.at<float>(i) = (float)kalman_points[(i - 1) / 2].y;
    }

    // Update the Measurement Matrix
    measurement += kf->measurementMatrix * state;
    kf->correct(measurement);
    std::cout << "face count " << faces.size() << std::endl;
}
