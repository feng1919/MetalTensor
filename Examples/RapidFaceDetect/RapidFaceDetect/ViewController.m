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

@interface ViewController ()

@property (nonatomic, strong) UILabel *labelFPS;

@property (nonatomic, strong) MetalImageView *metalView;
@property (nonatomic, strong) MetalImageVideoCamera *videoCamera;

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
    
    self.videoCamera = [[MetalImageVideoCamera alloc] initWithSessionPreset:AVCaptureSessionPreset1280x720 cameraPosition:AVCaptureDevicePositionBack];
    self.videoCamera.outputImageOrientation = UIInterfaceOrientationPortrait;
    self.videoCamera.horizontallyMirrorFrontFacingCamera = YES;
    self.videoCamera.horizontallyMirrorRearFacingCamera = NO;
    
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
    
    [self.videoCamera addTarget:self.metalView];

    [self.view bringSubviewToFront:self.labelFPS];
    
    [self.videoCamera performSelector:@selector(startCameraCapture) withObject:nil afterDelay:0.0f];
        
}

- (BOOL)prefersStatusBarHidden {
    return YES;
}

@end
