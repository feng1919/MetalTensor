//
//  MIPaddingItem.m
//  MetalTensor
//
//  Created by Feng Stone on 2019/11/6.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "MIPaddingItem.h"

MIPaddingItem *MPSPaddingTensorFlowSame = nil;
MIPaddingItem *MPSPaddingTensorFlowSameTransposee = nil;
MIPaddingItem *MPSPaddingValid = nil;
MIPaddingItem *MPSPaddingFull = nil;

@implementation MIPaddingItem

+ (void)initialize {
    if (self == [MIPaddingItem class]) {
        MPSPaddingTensorFlowSame = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_tfsame];
        MPSPaddingTensorFlowSameTransposee = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_tfsame_trans];
        MPSPaddingValid = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_valid];
        MPSPaddingFull = [[MIPaddingItem alloc] initWithPaddingMode:MTPaddingMode_full];
    }
}

MIPaddingItem *SharedPaddingItemWithMode(MTPaddingMode mode) {
    if (MPSPaddingTensorFlowSame == nil) {
        [MIPaddingItem load];
    }
    
    switch (mode) {
        case MTPaddingMode_tfsame:
            return MPSPaddingTensorFlowSame;
            break;
            
        case MTPaddingMode_tfsame_trans:
            return MPSPaddingTensorFlowSameTransposee;
            break;
            
        case MTPaddingMode_full:
            return MPSPaddingFull;
            break;
        
        case MTPaddingMode_valid:
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
            
        case MTPaddingMode_tfsame_trans:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodAddRemainderToTopLeft | MPSNNPaddingMethodSizeSame);
            
        case MTPaddingMode_valid:
            return (MPSNNPaddingMethodAlignCentered | MPSNNPaddingMethodSizeValidOnly);
            break;
            
        case MTPaddingMode_full:
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
            return @"MIPaddingItem( AlignCentered | AddRemainderToBottomRight | SizeSame)";
            break;
            
        case MTPaddingMode_tfsame_trans:
            return @"MIPaddingItem( AlignCentered | AddRemainderToTopLeft | SizeSame)";
            break;
            
        case MTPaddingMode_valid:
            return @"MIPaddingItem( AlignCentered | SizeValidOnly)";
            break;
            
        case MTPaddingMode_full:
            return @"MIPaddingItem( AlignCentered | SizeFull | AddRemainderToTopLeft | AddRemainderToBottomRight)";
            
        default:
            return @"MIPaddingItem( AlignCentered | AddRemainderToTopLeft | SizeSame)";
            break;
    }
}

- (NSString *)description {
    return self.label;
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeInt:_padding forKey:@"padding"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {
    if (self = [super init]) {
        _padding = [coder decodeIntForKey:@"padding"];
    }
    return self;
}

@end
