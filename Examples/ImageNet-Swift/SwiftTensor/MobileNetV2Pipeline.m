#import "MobileNetV2Pipeline.h"

#import <MetalImage/MetalImage.h>
#import <MetalTensor/FPSCounter.h>

#import "MobileNetV2.h"

@interface MobileNetV2Pipeline ()

@property (nonatomic, strong) MetalImageView *metalView;
@property (nonatomic, strong) MetalImageVideoCamera *videoCamera;
@property (nonatomic, strong) MICropFilter *cropFilter;
@property (nonatomic, strong) MetalImageFilter *resizeFilter;
@property (nonatomic, strong) MobileNetV2 *classifier;
@property (nonatomic, assign) BOOL started;

@end

@implementation MobileNetV2Pipeline

- (instancetype)initWithFrame:(CGRect)frame {
    if (self = [super init]) {
#if METAL_DEBUG
        _metalView = [[MetalImageDebugView alloc] initWithFrame:frame];
#else
        _metalView = [[MetalImageView alloc] initWithFrame:frame];
#endif
        _metalView.fillMode = kMetalImageFillModePreserveAspectRatio;
        _metalView.autoresizingMask = UIViewAutoresizingFlexibleWidth | UIViewAutoresizingFlexibleHeight;

        _videoCamera = [[MetalImageVideoCamera alloc] initWithSessionPreset:AVCaptureSessionPreset1920x1080
                                                              cameraPosition:AVCaptureDevicePositionBack];
        _videoCamera.outputImageOrientation = UIInterfaceOrientationPortrait;
        _videoCamera.horizontallyMirrorFrontFacingCamera = YES;
        _videoCamera.horizontallyMirrorRearFacingCamera = NO;

        _cropFilter = [[MICropFilter alloc] initWithCropRegion:CGRectMake(0, 210.0f / 960.0f, 1.0f, 1.0f - 420.0f / 960.0f)];
        _resizeFilter = [[MetalImageFilter alloc] init];

        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(handlePrediction:)
                                                     name:MOBILENET_PREDICTING_RESULT
                                                   object:nil];
    }

    return self;
}

- (UIView *)previewView {
    return self.metalView;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    [self stop];
}

- (void)start {
    if (self.started) {
        if (![self.videoCamera isCameraRunning]) {
            [self.videoCamera startCameraCapture];
        }
        return;
    }

    self.classifier = [[MobileNetV2 alloc] init];
    [self.classifier compile:[MetalDevice sharedMTLDevice]];
    [self.classifier loadWeights];

    self.resizeFilter.outputImageSize = [self.classifier inputSize];

    [self.videoCamera addTarget:self.cropFilter];
    [self.cropFilter addTarget:self.resizeFilter];
    [self.resizeFilter addTarget:self.classifier];
    [self.cropFilter addTarget:self.metalView];

    [self.videoCamera startCameraCapture];
    self.started = YES;
}

- (void)stop {
    if (![self.videoCamera isCameraRunning]) {
        return;
    }

    [self.videoCamera stopCameraCapture];
}

- (void)handlePrediction:(NSNotification *)notification {
    NSParameterAssert([notification.name isEqualToString:MOBILENET_PREDICTING_RESULT]);

    NSDictionary *result = (NSDictionary *)notification.object;
    NSArray<NSNumber *> *rates = result[@"RATES"];
    NSArray<NSString *> *labels = result[@"LABELS"];
    NSParameterAssert(rates.count == labels.count);

    float maxRate = 0.0f;
    NSString *predict = nil;
    for (NSInteger i = 0; i < rates.count; i++) {
        float rate = rates[i].floatValue;
        if (rate > maxRate) {
            maxRate = rate;
            predict = labels[i];
        }
    }

    if (!self.predictionHandler) {
        return;
    }

    int fps = [[FPSCounter sharedCounter] FPS];
    [[NSOperationQueue mainQueue] addOperationWithBlock:^{
        self.predictionHandler(predict, maxRate, fps);
    }];
}

@end
