//
//  ImageNet1000.m
//  MetalImage
//
//  Created by Feng Stone on 2019/5/24.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "ImageNet1000.h"

static ImageNet1000 *_classifier = nil;

@interface ImageNet1000() {
    
    NSArray<NSString *> *_labels;
    float *_rateBuffer;
}

@end

@implementation ImageNet1000

+ (void)initialize {
    if (self == [ImageNet1000 class]) {
        _classifier = [[ImageNet1000 alloc] init];
    }
}

+ (id)sharedInstance {
    return _classifier;
}

- (instancetype)init {
    if (self = [super init]) {
        
        NSString *txtFilePath = [[NSBundle mainBundle] pathForResource:@"synset_words" ofType:@"txt"];
        NSParameterAssert([[NSFileManager defaultManager] fileExistsAtPath:txtFilePath]);
        NSString *stringFromFile = [[NSString alloc] initWithContentsOfFile:txtFilePath encoding:NSUTF8StringEncoding error:nil];
        NSArray<NSString *> *l = [stringFromFile componentsSeparatedByString:@"\n"];
        NSMutableArray *lm = [NSMutableArray arrayWithCapacity:l.count];
        for (int i = 0; i < l.count; i++) {
            NSString *s = l[i];
            if ([s length] > 11) {
                [lm addObject:[s substringFromIndex:10]];
            }
        }
        _labels = [NSArray arrayWithArray:lm];
        NSParameterAssert(_labels.count > 0);
        _rateBuffer = malloc(_labels.count * sizeof(float));
        
    }
    return self;
}

- (float *)rateBuffer {
    return _rateBuffer;
}

- (void)dealloc {
    free(_rateBuffer);
}

typedef struct {
    int index;
    float rate;
}Rank;

- (NSDictionary *)rank5 {

    int count = (int)_labels.count;
    Rank *rankList = malloc(count * sizeof(Rank));
    Rank *rank5 = malloc(5 * sizeof(Rank));
    for (int i = 0; i < count; i++) {
        rankList[i].index = i;
        rankList[i].rate = _rateBuffer[i];
    }
    getTopRank(rankList, count, rank5, 5);
    NSArray *rates = @[@(rank5[4].rate), @(rank5[3].rate), @(rank5[2].rate), @(rank5[1].rate), @(rank5[0].rate)];
    NSArray *ls = @[_labels[rank5[4].index],
                    _labels[rank5[3].index],
                    _labels[rank5[2].index],
                    _labels[rank5[1].index],
                    _labels[rank5[0].index]];
    
    NSDictionary *result = @{@"RATES":rates, @"LABELS":ls};
    
    free(rank5);
    free(rankList);
    
    return result;
}

void getTopRank(Rank *rankBuffer,int count, Rank *result, int k) {
    assert(k<=count);
    for (int i = 0; i < k; i++) {
        result[i] = rankBuffer[i];
    }
    
    SortMinHeap(result, k);
    
    for (int i = k; i < count; i++) {
        if (result[0].rate < rankBuffer[i].rate) {
            result[0].rate = rankBuffer[i].rate;
            result[0].index = rankBuffer[i].index;
            SortMinHeap(result, k);
        }
    }
}

void SortMinHeap(Rank *rankHeap, int count) {
    
    Rank rankNon = {-1, -1};
    for (int i = count/2-1; i>=0; i--) {
        Rank r = rankHeap[i];
        Rank left = rankHeap[2*i+1];
        Rank right = (2*i+2<count)?rankHeap[2*i+2]:rankNon;
        if (r.rate > left.rate) {
            if (left.rate > right.rate) {
                rankHeap[i].index = right.index;
                rankHeap[i].rate = right.rate;
                rankHeap[2*i+2].index = r.index;
                rankHeap[2*i+2].rate = r.rate;
            }
            else {
                rankHeap[i].index = left.index;
                rankHeap[i].rate = left.rate;
                rankHeap[2*i+1].index = r.index;
                rankHeap[2*i+1].rate = r.rate;
            }
        }
        else if (r.rate > right.rate) {
            rankHeap[i].index = right.index;
            rankHeap[i].rate = right.rate;
            rankHeap[2*i+2].index = r.index;
            rankHeap[2*i+2].rate = r.rate;
        }
    }
}


@end
