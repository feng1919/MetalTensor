//
//  ViewController.m
//  MobileNetV2
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "ViewController.h"
#import <MetalImage/MetalImage.h>
#import <MetalTensor/FPSCounter.h>
#import <MetalTensor/MetalTensor.h>
#import "RapidFaceDetectFilter.h"

@interface ViewController ()

@property (nonatomic, strong) UILabel *labelFPS;

@property (nonatomic, strong) NSTimer *timer;
@property (nonatomic, strong) MetalImageView *metalView;
@property (nonatomic, strong) MetalImageVideoCamera *videoCamera;
@property (nonatomic, strong) RapidFaceDetectFilter *faceDetectFilter;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];

    CGRect bounds = self.view.bounds;
        
#if METAL_DEBUG
    self.metalView = [[MetalImageDebugView alloc] initWithFrame:bounds];
#else
    self.metalView = [[MetalImageView alloc] initWithFrame:bounds];
#endif
    self.metalView.fillMode = kMetalImageFillModePreserveAspectRatio;
    self.metalView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    
    [self.view addSubview:self.metalView];
    
    self.videoCamera = [[MetalImageVideoCamera alloc] initWithSessionPreset:AVCaptureSessionPreset1920x1080 cameraPosition:AVCaptureDevicePositionFront];
    self.videoCamera.outputImageOrientation = UIInterfaceOrientationPortrait;
    self.videoCamera.horizontallyMirrorFrontFacingCamera = YES;
    self.videoCamera.horizontallyMirrorRearFacingCamera = NO;
    
    MetalImageFilter *lens = [[MetalImageFilter alloc] init];
    [self.videoCamera addTarget:lens];
    
    CGFloat width = [UIScreen mainScreen].bounds.size.width;
    CGFloat height = [UIScreen mainScreen].bounds.size.height;
    MICropFilter *crop = [[MICropFilter alloc] initWithCropRegion:CGRectMake(0.0, (height-width)*0.5f/height, 1.0f, width/height)];
    [lens addTarget:crop];
    
    self.faceDetectFilter = [[RapidFaceDetectFilter alloc] init];
    [self.faceDetectFilter createNet];
    
    [crop addTarget:self.faceDetectFilter];
    [self.faceDetectFilter addTarget:self.metalView];
    
    self.labelFPS = [[UILabel alloc] initWithFrame:CGRectMake(0, CGRectGetHeight(bounds)-100, CGRectGetWidth(bounds), 50)];
    self.labelFPS.textColor = [UIColor whiteColor];
    self.labelFPS.font = [UIFont boldSystemFontOfSize:13];
    self.labelFPS.textAlignment = NSTextAlignmentCenter;
    self.labelFPS.autoresizingMask = (UIViewAutoresizingFlexibleWidth |
                                   UIViewAutoresizingFlexibleTopMargin);
    [self.view addSubview:self.labelFPS];
    
}

- (void)viewDidAppear:(BOOL)animated {
    [super viewDidAppear:animated];

    [self.view bringSubviewToFront:self.labelFPS];
    
    [self.videoCamera performSelector:@selector(startCameraCapture) withObject:nil afterDelay:0.0f];

    _timer = [NSTimer scheduledTimerWithTimeInterval:1.0f target:self selector:@selector(updateFPS) userInfo:nil repeats:YES];
}

- (void)updateFPS {
    [[NSOperationQueue mainQueue] addOperationWithBlock:^{
        self.labelFPS.text = [NSString stringWithFormat:@"%d FPS", [FPSCounter sharedCounter].FPS];
    }];
}

- (BOOL)prefersStatusBarHidden {
    return YES;
}

@end
