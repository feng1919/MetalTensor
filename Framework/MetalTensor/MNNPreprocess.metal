//
//  MIMeanAdjustKernel.metal
//  MetalImage
//
//  Created by Feng Stone on 2019/5/18.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include <metal_stdlib>

//constant half3 VGG_MEAN_SUBTRACT = half3(103.939f, 116.779f, 123.68f);
using namespace metal;

kernel void tensorflow_mean_nomarlization(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                                          texture2d<half, access::write> outTexture  [[ texture(1) ]],
                                          uint2 gid [[thread_position_in_grid]]) {
    half4 value = inTexture.read(gid);
    outTexture.write(half4(value.b*2.0h-1.0h,
                           value.g*2.0h-1.0h,
                           value.r*2.0h-1.0h,
                           1.0h), gid);
}

kernel void tensorflow_mean_norm_rgb(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                                     texture2d<half, access::write> outTexture  [[ texture(1) ]],
                                     uint2 gid [[thread_position_in_grid]]) {
    half4 value = inTexture.read(gid);
    outTexture.write(half4(value.r*2.0h-1.0h,
                           value.g*2.0h-1.0h,
                           value.b*2.0h-1.0h,
                           1.0h), gid);
}

kernel void tensorflow_mean_norm_rgba(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                                     texture2d<half, access::write> outTexture  [[ texture(1) ]],
                                     uint2 gid [[thread_position_in_grid]]) {
    half4 value = inTexture.read(gid);
    outTexture.write(half4(value.r*2.0h-1.0h,
                           value.g*2.0h-1.0h,
                           value.b*2.0h-1.0h,
                           1.0h), gid);
}

kernel void tensorflow_mean_nomarlization_paddingZeroAtBottomAndRight(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                                                                      texture2d<half, access::write> outTexture  [[ texture(1) ]],
                                                                      device uint *size                         [[buffer(0)]],
                                                                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x==size[0] || gid.y==size[0]) {
        outTexture.write(half4(0.0h), gid);
        return;
    }
    
    half4 value = inTexture.read(gid);
    outTexture.write(half4(value.b*2.0h-1.0h,
                           value.g*2.0h-1.0h,
                           value.r*2.0h-1.0h,
                           1.0h), gid);
}

kernel void caffe_mean(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                       texture2d<half, access::write> outTexture  [[ texture(1) ]],
                       uint2 gid [[thread_position_in_grid]]) {
    half4 value = inTexture.read(gid);
    outTexture.write(half4(value.b*255.0h-103.939h,
                           value.g*255.0h-116.779h,
                           value.r*255.0h-123.68h,
                           1.0h), gid);
}

kernel void caffe_scale_mean(texture2d<half, access::read> inTexture  [[ texture(0) ]],
                             texture2d<half, access::write> outTexture  [[ texture(1) ]],
                             uint2 gid [[thread_position_in_grid]]) {
    // Subtract mean values, scale by 0.017, convert to BGR.
    const float4 means = float4(123.68f, 116.78f, 103.94f, 0.0f);
    const float4 inColor = (float4(inTexture.read(gid)) * 255.0f - means) * 0.017f;
    outTexture.write(half4(inColor.z, inColor.y, inColor.x, 0.0f), gid);
}
