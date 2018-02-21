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

#include <iostream>

#include "Face.h"

@interface ViewController ()
<AVCaptureVideoDataOutputSampleBufferDelegate>

@end

@implementation ViewController
{
    AVCaptureSession *session;
    dispatch_queue_t faceQueue;
    AVSampleBufferDisplayLayer *displayLayer;
    NSMutableArray<AVMetadataFaceObject *>* faceObjects;
    std::vector<Face> faces;
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

    AVCaptureDevice *captureDevice;
    for (AVCaptureDevice *device in [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo]) {
        if (device.position == AVCaptureDevicePositionFront) {
            captureDevice = device;
        }
    }

    if(captureDevice == nil) {
        [NSException raise:@"" format:@"AVCaptureDevicePositionBack not found"];
    }

    session = [[AVCaptureSession alloc] init];

    [session beginConfiguration];
    session.sessionPreset = AVCaptureSessionPreset352x288;

    AVCaptureDeviceInput *deviceInput = [AVCaptureDeviceInput deviceInputWithDevice:captureDevice error:&error];
    if (error) {
        [NSException raise:@"" format:@"AVCaptureDeviceInput not found"];
    }
    [session addInput:deviceInput];

    AVCaptureVideoDataOutput *videoOutput = [[AVCaptureVideoDataOutput alloc] init];
    videoOutput.videoSettings = @{(id)kCVPixelBufferPixelFormatTypeKey : @(kCVPixelFormatType_32BGRA) };
    [videoOutput setSampleBufferDelegate:self queue:dispatch_get_main_queue()];
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
    faces.clear();

    CVImageBufferRef pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);

    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    int width = static_cast<int>(CVPixelBufferGetWidth(pixelBuffer));
    int height = static_cast<int>(CVPixelBufferGetHeight(pixelBuffer));
    int bytesPerRow = static_cast<int>(CVPixelBufferGetBytesPerRow(pixelBuffer));
    unsigned char *baseBuffer = (unsigned char *)CVPixelBufferGetBaseAddress(pixelBuffer);

    // Prepare image for mtcnn
    cv::Mat img = cv::Mat(height, width, CV_8UC4, baseBuffer, bytesPerRow);
    if(!img.data){
        std::cout << "Reading video failed" << std::endl;
        return;
    }

    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);

    // Process

    // Draw
    [self drawResult:img];

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

- (void)drawResult:(cv::Mat& )image {
    for (const auto& face : faces) {
        cv::rectangle(image, face.rect, cv::Scalar(255));
        for (int i = 0; i < face.landmarks.size(); ++i)
            cv::circle(image, face.landmarks[i], 1, cv::Scalar(255, 0, 0));
    }
}
@end
