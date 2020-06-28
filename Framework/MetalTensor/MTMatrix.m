//
//  MTMatrix.m
//  MetalTensor
//
//  Created by Feng Stone on 2020/2/20.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import "MTMatrix.h"
#import "MTTensorCache.h"

@interface MTMatrix() {
    MPSTemporaryMatrix *_matrix;
}

@end

@implementation MTMatrix
@synthesize referenceCountingEnable = _referenceCountingEnable;
@synthesize referenceCounting = _referenceCounting;
@synthesize poolIdentifier = _poolIdentifier;

- (instancetype)init{
    NSAssert(NO, @"Invalid initialize function.");
    return nil;
}

- (instancetype)initWithRows:(int)rows columns:(int)columns {
    return [self initWithRows:rows columns:columns dataType:MPSDataTypeFloat16];
}

- (instancetype)initWithRows:(int)rows columns:(int)columns dataType:(MPSDataType)dataType {
    if (self = [super init]) {
        _dataType = dataType;
        _rows = rows;
        _columns = columns;
        _referenceCounting = 0;
        _matrixDescriptor = MatrixDescriptor(rows, columns, _dataType);
        _referenceCountingEnable = YES;
    }
    return self;
}


- (void)dealloc {
    
    NSAssert(_referenceCountingEnable == NO || _referenceCounting == 0, @"Unexpected dealloc...");
    
    self.matrix = nil;
    
#if DEBUG
    if (_referenceCountingEnable) {
        NSLog(@"Temporary matrix dealloc: %dx%d", _rows, _columns);
    }
#endif
}

- (MPSMatrix *)content {
    return _matrix;
}

- (void)setMatrix:(MPSTemporaryMatrix *)matrix {
    NSAssert(matrix==nil||[matrix isKindOfClass:[MPSTemporaryMatrix class]], @"Invalid matrix type...");
    @synchronized (self) {
        if (_matrix.readCount > 0) {
            _matrix.readCount = 0;
        }
        _matrix = matrix;
    }
}

#pragma mark - MTResourcce delegate

- (void)lock {
    if (_referenceCountingEnable) {
        _referenceCounting++;
    }
}

- (void)unlock {
    if (_referenceCountingEnable) {
        NSAssert(_referenceCounting > 0, @"Tried to overrelease a temporary image.");
        _referenceCounting--;
        if (_referenceCounting < 1) {
            //        if ([_image isKindOfClass:[MPSTemporaryImage class]]) {
            //            [(MPSTemporaryImage *)_image setReadCount:0];
            //        }
            //        self.image = nil;
            [[MTTensorCache sharedCache] cacheResource:self];
        }
    }
}

- (int)referenceCounting {
    return _referenceCounting;
}

- (NSString *)reuseIdentifier {
    return KeyForMatrixType(self.rows, self.columns, self.dataType);
}

- (void)newContentOnCommandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    @synchronized (self) {
        if (!_matrix) {
//        _image.readCount = 0;
            _matrix = [MPSTemporaryMatrix temporaryMatrixWithCommandBuffer:commandBuffer matrixDescriptor:_matrixDescriptor];
            _matrix.readCount = NSIntegerMax;
        }
    }
}

- (void)deleteContent {
    self.matrix = nil;
}

@end

NSString *KeyForMatrixType(int rows, int columns, MPSDataType dataType) {
    return [NSString stringWithFormat:@"<matrix: %dx%d - float%d>", rows, columns, dataType==MPSDataTypeFloat16?16:32];
}

MPSMatrixDescriptor *MatrixDescriptor(int row, int column, MPSDataType dataType) {
    assert(row>0 && column>0);
    int size = dataType == MPSDataTypeFloat16?sizeof(float16_t):sizeof(float32_t);
    MPSMatrixDescriptor *desc = [MPSMatrixDescriptor matrixDescriptorWithRows:row columns:column rowBytes:column*size dataType:dataType];
//    desc.storageMode = MTLStorageModePrivate;
    return desc;
}
