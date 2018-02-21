//
//  ViewController.m
//  FaceTrackerSampleApp
//
//  Created by Chihiro on 2018/02/21.
//  Copyright © 2018 Chihiro. All rights reserved.
//

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#import "ViewController.h"

#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#import <UIKit/UIKit.h>
#import <AVFoundation/AVFoundation.h>
#import <CoreMedia/CoreMedia.h>

@interface ViewController ()
<AVCaptureVideoDataOutputSampleBufferDelegate>

@end

@implementation ViewController
{
    AVCaptureSession *session;
    dispatch_queue_t faceQueue;

    AVSampleBufferDisplayLayer *displayLayer;

    NSMutableArray<AVMetadataFaceObject *>* faceObjects;
}

-(void)dealloc {
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    displayLayer = [AVSampleBufferDisplayLayer new];
    [self initCamera];

}

- (void)viewDidAppear:(BOOL)animated
{
    [super viewDidAppear:animated];
    displayLayer.frame = self.view.bounds;
    [self.view.layer addSublayer:displayLayer];
    [self.view layoutIfNeeded];
}

#pragma mark - Camera

- (void)initCamera
{
    NSError *error;

    faceQueue = dispatch_queue_create("com.f.faceQueue", DISPATCH_QUEUE_SERIAL);

    AVCaptureDevice *captureDevice;  // デバイスを取得
    for (AVCaptureDevice *device in [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]) {
        if (device.position == AVCaptureDevicePositionFront) {
            captureDevice = device;
        }
    }

    if(captureDevice == nil) {
        [NSException raise:@"" format:@"AVCaptureDevicePositionBack not found"];
    }

    session = [[AVCaptureSession alloc] init];  // セッションを作成

    [session beginConfiguration];  // セッションの設定
    session.sessionPreset = AVCaptureSessionPreset352x288;

    AVCaptureDeviceInput *deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:captureDevice error:&error];  // インプットを作成して追加
    if (error) {
        [NSException raise:@"" format:@"AVCaptureDeviceInput not found"];
    }
    [session addInput:deviceInput];

    AVCaptureVideoDataOutput *videoOutput = [[AVCaptureVideoDataOutput alloc] init];  // アウトプットを作成して追加
    videoOutput.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) }; // 設定
    [videoOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()]; // 出力先
    [session addOutput:videoOutput];

    [session commitConfiguration];

    // カメラからの入力を開始
    [session startRunning];

    // 入力の方向を変える
    for(AVCaptureConnection *connection in videoOutput.connections)
    {
        if(connection.supportsVideoOrientation)
        {
            connection.videoOrientation = AVCaptureVideoOrientationPortrait;
        }
    }
}

#pragma mark - AVCaptureVideoDataOutputSampleBufferDelegate

// カメラの出力の生データ
- (void)captureOutput:(AVCaptureOutput *)captureOutput didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer fromConnection:(AVCaptureConnection *)connection
{
    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    int width = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
    int height = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
    int bytesPerRow = static_cast<int>(CVPixelBufferGetBytesPerRow(pixelBuffer));
    unsigned char *baseBuffer = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    // Prepare image for mtcnn
    cv::Mat img = cv::Mat(height, width, CV_8UC4, baseBuffer, bytesPerRow); //put buffer in open cv, no memory copied
    if(!img.data){
        cout<<"Reading video failed"<<endl;
        return;
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    cv::cvtColor(img, img, CV_BGRA2BGR);
    if(!img.data) cout<<"Reading video failed!"<<endl;


    // Prepare mtcnn face detection
    vector<cv::Rect> rects;
    vector<float> confidences;
    std::vector<std::vector<cv::Point>> alignment;

    NSDate *startDate = [NSDate new];

    std::vector<MtcnnResult> faceDetectionResults = faceDetector.run(img, FaceDetector::WOWO);

    NSTimeInterval interval = [[NSDate date] timeIntervalSinceDate:startDate];
    NSLog(@"inerval: %fms", interval * 1000 );

    // Initialize measurement points
    static std::vector<cv::Point2f> kalman_points(68, cv::Point2f(0.0, 0.0));

    // Initialize prediction points
    static std::vector<cv::Point2f> predict_points(68, cv::Point2f(0.0, 0.0));

    // Kalman Filter Setup (68 Points Test)
    static const int stateNum = 68 * 4;
    static const int measureNum = 68 * 2;

    KalmanFilter KF(stateNum, measureNum, 0);
    Mat state(stateNum, 1, CV_32FC1);
    Mat processNoise(stateNum, 1, CV_32F);
    Mat measurement = Mat::zeros(measureNum, 1, CV_32F);
    cv::Mat prevgray, grayImage;
    static std::vector<cv::Point2f> prevTrackPts(68, cv::Point2f(0, 0));
    static std::vector<cv::Point2f> nextTrackPts;
    cv::cvtColor(img, grayImage, CV_BGR2GRAY);
    int flag = -1;
    for (const auto& result : faceDetectionResults) {
        cv::Rect rect = result.bb;
        dlib::rectangle r(rect.tl().x, rect.tl().y, rect.br().x, rect.br().y);
        dlib::full_object_detection shape = landmarkDetector(dlib::cv_image<uint8_t>(grayImage), r);

        if (flag == -1) {
            cvtColor(img, prevgray, CV_BGR2GRAY);
            const dlib::full_object_detection& d = shape;
            for (int i = 0; i < d.num_parts(); i++) {
                prevTrackPts[i].x = d.part(i).x();
                prevTrackPts[i].y = d.part(i).y();
            }
            flag = 1;
        }
        // Update Kalman Filter Points
        const dlib::full_object_detection& d = shape;
        for (int i = 0; i < d.num_parts(); i++) {
            kalman_points[i].x = d.part(i).x();
            kalman_points[i].y = d.part(i).y();
        }

        // Kalman Prediction
        Mat prediction = KF.predict();

        for (int i = 0; i < 68; i++) {
            predict_points[i].x = prediction.at<float>(i * 2);
            predict_points[i].y = prediction.at<float>(i * 2 + 1);
        }

        static bool redetected = true;
        if (prevgray.data) {
            std::vector<uchar> status;
            std::vector<float> err;
            calcOpticalFlowPyrLK(prevgray, prevgray, prevTrackPts, nextTrackPts, status, err);
            std::cout << "variance:" <<calDistanceDiff(prevTrackPts, nextTrackPts) << std::endl;
            // if the face is moving so fast, use dlib to detect the face
            double diff = calDistanceDiff(prevTrackPts, nextTrackPts);
            if (diff > 1.0) {
                const dlib::full_object_detection& d = shape;
                std::cout<< "DLIB" << std::endl;
                for (int i = 0; i < d.num_parts(); i++) {
                    cv::circle(img, cv::Point2f(d.part(i).x(), d.part(i).y()), 1, cv::Scalar(0, 0, 255), -1);
                    nextTrackPts[i].x = d.part(i).x();
                    nextTrackPts[i].y = d.part(i).y();
                }
            } else if (diff <= 1.0 && diff > 0.005){
                // In this case, use Optical Flow
                std::cout<< "Optical Flow" << std::endl;
                for (int i = 0; i < nextTrackPts.size(); i++) {
                    cv::circle(img, nextTrackPts[i], 1, cv::Scalar(255, 0, 0), -1);
                }
            } else {
                // In this case, use Kalman Filter
                std::cout<< "Kalman Filter" << std::endl;
                for (int i = 0; i < predict_points.size(); i++) {
                    cv::circle(img, predict_points[i], 1, cv::Scalar(0, 255, 0), -1);
                    nextTrackPts[i].x = predict_points[i].x;
                    nextTrackPts[i].y = predict_points[i].y;
                }
                redetected = false;
            }
        } else {
            redetected = true;
        }
        std::swap(prevTrackPts, nextTrackPts);
        std::swap(prevgray, grayImage);

        // draw?
        cv::rectangle(img, rect, cv::Scalar(255, 127, 255), 1);
        for (int i = 0; i < shape.num_parts(); ++i)
            cv::circle(img, cv::Point((int)shape.part(i).x(), (int)shape.part(i).y()), 1, Scalar(127,255,191));
    }

    // BGRイメージをRGBの配列に戻してpixel_bufferに戻す
    long location = 0;
    uint8_t* pixel_ptr = (uint8_t*)img.data;
    int cn = img.channels();
    cv::Scalar_<uint8_t> rgb_pixel;
    for(int i = 0; i < img.rows; i++) {
        for(int j = 0; j < img.cols; j++) {
            long bufferLocation = location * 4; // 4: RBGA
            rgb_pixel.val[0] = pixel_ptr[i*img.cols*cn + j*cn + 2]; // R
            rgb_pixel.val[1] = pixel_ptr[i*img.cols*cn + j*cn + 1]; // G
            rgb_pixel.val[2] = pixel_ptr[i*img.cols*cn + j*cn + 0]; // B
            baseBuffer[bufferLocation] = rgb_pixel.val[2];
            baseBuffer[bufferLocation + 1] = rgb_pixel.val[1];
            baseBuffer[bufferLocation + 2] = rgb_pixel.val[0];
            location++;
        }
    }

    [displayLayer enqueueSampleBuffer:sampleBuffer];
}
@end
