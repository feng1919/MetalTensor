//
//  MTMatrix.h
//  MetalTensor
//
//  Created by Feng Stone on 2020/2/20.
//  Copyright © 2020 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MPSImage+Extension.h"
#import "metal_tensor_structures.h"
#import "MTResourceProtocol.h"

NS_ASSUME_NONNULL_BEGIN

@interface MTMatrix : NSObject <MTResource>

@property (nonatomic, readonly) MPSMatrixDescriptor *matrixDescriptor;
@property (nonatomic, readonly) MPSDataType dataType;
@property (nonatomic, readonly) int rows;
@property (nonatomic, readonly) int columns;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithRows:(int)rows columns:(int)columns;
- (instancetype)initWithRows:(int)rows columns:(int)columns dataType:(MPSDataType)dataType NS_DESIGNATED_INITIALIZER;

- (MPSMatrix *)content;

@end

MPSMatrixDescriptor *MatrixDescriptor(int row, int column, MPSDataType dataType);
NSString *KeyForMatrixType(int rows, int columns, MPSDataType dataType);
typedef MTMatrix * MetalMatrix;

NS_ASSUME_NONNULL_END
