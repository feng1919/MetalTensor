//
//  MIPaddingItem.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/11/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIPaddingItem.h"

MIPaddingItem *MPSPaddingTensorFlowSame = nil;
MIPaddingItem *MPSPaddingValid = nil;
MIPaddingItem *MPSPaddingFull = nil;

@implementation MIPaddingItem

+ (void)initialize {
    if (self == [MIPaddingItem class]) {
        MPSPaddingTensorFlowSame = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_tfsame];
        MPSPaddingValid = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_tfvalid];
        MPSPaddingFull = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_tffull];
    }
}

MIPaddingItem *SharedPaddingItemWithMode(MTPaddingMode mode) {
    switch (mode) {
        case MTPaddingMode_tfsame:
            return MPSPaddingTensorFlowSame;
            break;
            
        case MTPaddingMode_tffull:
            return MPSPaddingFull;
            break;
        
        case MTPaddingMode_tfvalid:
            return MPSPaddingValid;
            break;
            
        default:
            assert(0); //Unsupportted padding mode
            return nil;
            break;
    }
}

- (instancetype)initWithPaddingMode:(MTPaddingMode)mode {
    if (self = [super init]) {
        _padding = mode;
    }
    return self;
}

- (MPSNNPaddingMethod)paddingMethod {
    switch (_padding) {
        case MTPaddingMode_tfsame:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodAddRemainderToBottomRight | MPSNNPaddingMethodSizeSame);
            break;
            
        case MTPaddingMode_tfvalid:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeValidOnly);
            break;
            
        case MTPaddingMode_tffull:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeFull |
                    MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodAddRemainderToBottomRight);
            break;
            
        default:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodSizeSame);
            break;
    };
}

- (NSString *)label {
    switch (_padding) {
        case MTPaddingMode_tfsame:
            return @"AlignCentered | AddRemainderToBottomRight | SizeSame";
            break;
            
        case MTPaddingMode_tfvalid:
            return @"AlignCentered | SizeValidOnly)";
            break;
            
        case MTPaddingMode_tffull:
            return @"AlignCentered | SizeFull | AddRemainderToTopLeft | AddRemainderToBottomRight)";
            
        default:
            return @"AlignCentered | AddRemainderToTopLeft | SizeSame";
            break;
    }
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeInt:_padding forKey:@"padding"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {
    if (self = [super init]) {
        self->_padding = [coder decodeIntForKey:@"padding"];
    }
    return self;
}

@end
