#import <XCTest/XCTest.h>
#import <MetalTensor/NSString+Extension.h>
#import <MetalTensor/MetalTensorLayerDescriptor.h>
#import <MetalTensor/metal_tensor_structures.h>
#import <MetalTensor/numpy.h>

@interface MetalTensorLogicTests : XCTestCase
@end

@implementation MetalTensorLogicTests

- (void)testRemoveBlankSpacesRecursively {
    NSDictionary *source = @{
        @"name": @" mobile net v2 ",
        @"config": @{
            @"shape": @" 1, 2, 3 ",
        },
        @"items": @[
            @" alpha beta ",
            @{@"weight": @" conv 1 "},
            @3,
        ],
        @"number": @1,
    };
    
    NSDictionary *result = [source removeBlankSpaces];
    NSDictionary *config = result[@"config"];
    NSArray *items = result[@"items"];
    NSDictionary *weightInfo = items[1];
    
    XCTAssertEqualObjects(result[@"name"], @"mobilenetv2");
    XCTAssertEqualObjects(config[@"shape"], @"1,2,3");
    XCTAssertEqualObjects(items[0], @"alphabeta");
    XCTAssertEqualObjects(weightInfo[@"weight"], @"conv1");
    XCTAssertEqualObjects(items[2], @3);
}

- (void)testDictionaryWithContentsOfFileRemovingBlankSpaces {
    NSDictionary *source = @{
        @"inputs": @" 1, 2, 3 ; 4, 5, 6 ",
        @"targets": @" layer1, layer2 ",
    };
    NSString *path = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSUUID UUID].UUIDString];
    
    XCTAssertTrue([source writeToFile:path atomically:YES]);
    
    NSDictionary *result = [NSDictionary dictionaryWithContentsOfFile:path removingBlankSpaces:YES];
    [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
    
    XCTAssertEqualObjects(result[@"inputs"], @"1,2,3;4,5,6");
    XCTAssertEqualObjects(result[@"targets"], @"layer1,layer2");
}

- (void)testDescriptorFactoriesAndBaseParsing {
    XCTAssertEqual(DescriptorWithType(nil), [MIConvolutionLayerDescriptor class]);
    XCTAssertEqual(DescriptorWithType(@"dense"), [MIFullyConnectedLayerDescriptor class]);
    XCTAssertEqual(LayerWithType(@"reshape"), [MIReshapeLayer class]);
    XCTAssertEqual(LayerWithType(@"output"), [MetalTensorOutputLayer class]);
    
    NSDictionary *dictionary = [@{
        @"inputs": @" 2, 3, 4 ; 5, 6, 7 ",
        @"output": @" 8, 9, 10 ",
        @"targets": @" layer_a, layer_b ",
        @"indices": @" 0, 1 ",
        @"type": @" output ",
        @"backward": @YES,
    } removeBlankSpaces];
    
    MetalTensorLayerDescriptor *descriptor = [[MetalTensorLayerDescriptor alloc] initWithDictionary:dictionary];
    DataShape *inputShapes = [descriptor inputShapeRef];
    
    XCTAssertEqualObjects(descriptor.type, @"output");
    XCTAssertEqual(descriptor.n_inputs, 2);
    XCTAssertEqual(inputShapes[0].row, 2);
    XCTAssertEqual(inputShapes[0].column, 3);
    XCTAssertEqual(inputShapes[0].depth, 4);
    XCTAssertEqual(inputShapes[1].row, 5);
    XCTAssertEqual(inputShapes[1].column, 6);
    XCTAssertEqual(inputShapes[1].depth, 7);
    XCTAssertEqual(descriptor.outputShape.row, 8);
    XCTAssertEqual(descriptor.outputShape.column, 9);
    XCTAssertEqual(descriptor.outputShape.depth, 10);
    XCTAssertEqualObjects(descriptor.targets, (@[@"layer_a", @"layer_b"]));
    XCTAssertEqualObjects(descriptor.targetIndices, (@[@"0", @"1"]));
    XCTAssertTrue(descriptor.needBackward);
}

- (void)testConvolutionDescriptorDefaults {
    NSDictionary *dictionary = [@{
        @"inputs": @" 4, 4, 3 ",
        @"kernel": @" 3 ",
        @"filters": @" 8 ",
        @"weight": @" conv_0 ",
    } removeBlankSpaces];
    
    MIConvolutionLayerDescriptor *descriptor = [[MIConvolutionLayerDescriptor alloc] initWithDictionary:dictionary];
    
    XCTAssertEqualObjects(descriptor.type, @"convolution");
    XCTAssertEqual(descriptor.kernelShape.row, 3);
    XCTAssertEqual(descriptor.kernelShape.column, 3);
    XCTAssertEqual(descriptor.kernelShape.depth, 3);
    XCTAssertEqual(descriptor.kernelShape.filters, 8);
    XCTAssertEqual(descriptor.kernelShape.stride, 1);
    XCTAssertEqual(descriptor.padding, MTPaddingMode_tfsame);
    XCTAssertEqual(descriptor.offset.x, conv_offset(3, 1, MTPaddingMode_tfsame));
    XCTAssertEqual(descriptor.offset.y, conv_offset(3, 1, MTPaddingMode_tfsame));
    XCTAssertFalse(descriptor.depthWise);
    XCTAssertEqualObjects(descriptor.weight, @"conv_0");
    XCTAssertEqual(descriptor.weightRange.location, NSNotFound);
    XCTAssertEqual(descriptor.weightRange.length, 0);
}

- (void)testShapeUtilities {
    DataShape shape = DataShapeMake(2, 3, 4);
    int row = -1;
    int column = 2;
    int depth = 3;
    
    XCTAssertEqual(Reshape1(&shape, &row, &column, &depth), 0);
    XCTAssertEqual(row, 4);
    XCTAssertEqual(shape.row, 4);
    XCTAssertEqual(shape.column, 2);
    XCTAssertEqual(shape.depth, 3);
    
    DataShape shapes[] = {
        DataShapeMake(1, 2, 3),
        DataShapeMake(4, 1, 5),
    };
    int offsets[2] = {0};
    DataShape concatenated = ConcatenateShapes(shapes, 2, offsets, true);
    
    XCTAssertEqual(offsets[0], 0);
    XCTAssertEqual(offsets[1], 4);
    XCTAssertEqual(concatenated.row, 4);
    XCTAssertEqual(concatenated.column, 2);
    XCTAssertEqual(concatenated.depth, 12);
    XCTAssertEqual(conv_output_length(5, 3, 2, MTPaddingMode_valid), 2);
    XCTAssertEqual(conv_output_length(5, 3, 2, MTPaddingMode_tfsame), 3);
}

- (void)testNumpyHelpers {
    float values[3] = {0.1f, 0.9f, 0.2f};
    XCTAssertEqual(argmax(values, 3), 1);
    
    float softmaxValues[3] = {1.0f, 2.0f, 3.0f};
    soft_max(softmaxValues, 3);
    float sum = softmaxValues[0] + softmaxValues[1] + softmaxValues[2];
    
    XCTAssertEqualWithAccuracy(sum, 1.0f, 0.0001f);
    XCTAssertTrue(softmaxValues[2] > softmaxValues[1]);
    XCTAssertTrue(softmaxValues[1] > softmaxValues[0]);
    
    float line[5] = {0};
    float step = 0.0f;
    XCTAssertEqual(linspace(0.0f, 1.0f, 5, true, line, &step), 5);
    XCTAssertEqualWithAccuracy(step, 0.25f, 0.0001f);
    XCTAssertEqualWithAccuracy(line[0], 0.0f, 0.0001f);
    XCTAssertEqualWithAccuracy(line[4], 1.0f, 0.0001f);
}

@end
