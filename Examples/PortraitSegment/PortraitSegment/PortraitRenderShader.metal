//
//  PortraitRender.metal
//  MetalTensorDemo
//
//  Created by Feng Stone on 2019/9/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct VertexIO
{
    float4 position [[position]];
    float2 textureCoordinate [[user(texturecoord)]];
};

fragment half4 fragment_portraitRender(VertexIO         inFrag     [[ stage_in ]],
                                       texture2d<half> texture1     [[ texture(0) ]],   // portrait image
                                       texture2d<half> texture2     [[ texture(1) ]])   // mask
{
    constexpr sampler quadSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    
    half4 textureColor = texture1.sample(quadSampler, inFrag.textureCoordinate);
    half4 textureColor2 = texture2.sample(quadSampler, inFrag.textureCoordinate);
    const half3 bg_color = half3(20.h/255.h, 40.h/255.h, 80.h/ 255.h);
    
    return half4(mix(bg_color, textureColor.rgb, textureColor2.r), 1.0h);
}
