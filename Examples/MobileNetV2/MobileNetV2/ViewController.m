//
//  ViewController.m
//  MobileNetV2
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "ViewController.h"
#import <MetalImage/MetalImage.h>
#import "MobileNetV2.h"
#import <MetalTensor/FPSCounter.h>

@interface ViewController ()

@property (nonatomic, strong) UILabel *label;
@property (nonatomic, strong) UILabel *labelFPS;

@property (nonatomic, strong) MetalImageView *metalView;
@property (nonatomic, strong) MetalImageVideoCamera *videoCamera;
@property (nonatomic, strong) MobileNetV2 *mobilenetv2;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(mobilenetv2PredictDidFinish:)
                                                 name:MOBILENET_PREDICTING_RESULT object:nil];

    CGRect bounds = self.view.bounds;
        
#if METAL_DEBUG
    self.metalView = [[MetalImageDebugView alloc] initWithFrame:bounds];
#else
    self.metalView = [[MetalImageView alloc] initWithFrame:bounds];
#endif
    self.metalView.fillMode = kMetalImageFillModePreserveAspectRatio;
    self.metalView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;
    
    [self.view addSubview:self.metalView];
    
    self.videoCamera = [[MetalImageVideoCamera alloc] initWithSessionPreset:AVCaptureSessionPreset1920x1080 cameraPosition:AVCaptureDevicePositionBack];
    self.videoCamera.outputImageOrientation = UIInterfaceOrientationPortrait;
    self.videoCamera.horizontallyMirrorFrontFacingCamera = YES;
    self.videoCamera.horizontallyMirrorRearFacingCamera = NO;
    
    self.label = [[UILabel alloc] initWithFrame:CGRectMake(0, 40, CGRectGetWidth(bounds), 60)];
    self.label.textColor = [UIColor whiteColor];
    self.label.font = [UIFont boldSystemFontOfSize:15];
    self.label.textAlignment = NSTextAlignmentCenter;
    self.label.numberOfLines = 2;
    self.label.autoresizingMask = (UIViewAutoresizingFlexibleWidth |
                                   UIViewAutoresizingFlexibleBottomMargin);
    [self.view addSubview:self.label];
    
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
    
    MICropFilter *cropFilter = [[MICropFilter alloc] initWithCropRegion:CGRectMake(0, 210.0f/960.0f, 1.0f, 1.0f-420.0f/960.0f)];

    self.mobilenetv2 = [[MobileNetV2 alloc] init];
    [self.mobilenetv2 compile:[MetalDevice sharedMTLDevice]];
    [self.mobilenetv2 loadWeights];

    MetalImageFilter *resize = [[MetalImageFilter alloc] init];
    resize.outputImageSize = [self.mobilenetv2 inputSize];

    [self.videoCamera addTarget:cropFilter];
    [cropFilter addTarget:resize];
    [resize addTarget:self.mobilenetv2];
    [cropFilter addTarget:self.metalView];

    [self.view bringSubviewToFront:self.label];
    [self.view bringSubviewToFront:self.labelFPS];
    
    [self.videoCamera performSelector:@selector(startCameraCapture) withObject:nil afterDelay:0.0f];
        
}

- (BOOL)prefersStatusBarHidden {
    return YES;
}

#pragma mark - MobilenetV2 Notification

- (void)mobilenetv2PredictDidFinish:(NSNotification *)note {
    
    NSParameterAssert([note.name isEqualToString:MOBILENET_PREDICTING_RESULT]);
    
    [[NSOperationQueue mainQueue] addOperationWithBlock:^{

        self.labelFPS.text = [NSString stringWithFormat:@"%d FPS", [FPSCounter sharedCounter].FPS];
        
        NSDictionary *result = (NSDictionary *)[note object];
        NSArray *rates = result[@"RATES"];
        NSArray *labels = result[@"LABELS"];
        NSParameterAssert(rates.count == labels.count);
        
        float max_rate = 0.0f;
        NSString *predict = nil;
        for (int i = 0; i < rates.count; i++) {
            if ([rates[i] floatValue] > max_rate) {
                max_rate = [rates[i] floatValue];
                predict = labels[i];
            }
        }
        
        self.label.text = [NSString stringWithFormat:@"%@ (%0.f%%)", predict, max_rate*100.0f];
    }];
}

@end
