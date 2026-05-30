#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

typedef void (^MobileNetV2PredictionHandler)(NSString * _Nullable label, float confidence, int fps);

@interface MobileNetV2Pipeline : NSObject

@property (nonatomic, strong, readonly) UIView *previewView;
@property (nonatomic, copy, nullable) MobileNetV2PredictionHandler predictionHandler;

- (instancetype)initWithFrame:(CGRect)frame NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

- (void)start;
- (void)stop;

@end

NS_ASSUME_NONNULL_END
