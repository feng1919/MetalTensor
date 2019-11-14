//
//  ObjectDetectNetRender.metal
//  MetalImage
//
//  Created by Feng Stone on 2019/6/27.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

struct VertexIO
{
    float4 position [[position]];
    float2 textureCoordinate [[user(texturecoord)]];
};


fragment half4 fragment_object_detect_net_render(VertexIO         inFrag    [[ stage_in ]],
                                                 texture2d<half>  tex2D     [[ texture(0) ]],
                                                 constant packed_float4 *minmax [[ buffer(0) ]],
                                                 constant int &count        [[ buffer(1) ]])
{
    constexpr sampler quadSampler(coord::normalized, filter::linear, address::clamp_to_edge);
    half4 inColor = tex2D.sample(quadSampler, inFrag.textureCoordinate);
    float x = inFrag.textureCoordinate.x;
    float y = inFrag.textureCoordinate.y;

    half4 white = half4(1.0h);
    for (int i = 0; i < count; i++)
    {
        if (x>minmax[i][0] && x<minmax[i][1] && y>minmax[i][2] && y<minmax[i][3]) {
            return mix(inColor, white, 0.5h);
        }
    }
    
    return inColor;
}
