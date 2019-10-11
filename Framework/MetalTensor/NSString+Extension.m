//
//  NSString+Extension.m
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "NSString+Extension.h"

@implementation NSString (Extension)

- (NSString *)removeBlankSpace {
    return [self stringByReplacingOccurrencesOfString:@" " withString:@""];
}

- (NSArray *)nonEmptyComponentsSeparatedByString:(NSString *)string {
    NSArray *array = [self componentsSeparatedByString:string];
    return [array removeBlankSpaces];
}

@end

@implementation NSDictionary (blankSpaces)

+ (NSDictionary *)dictionaryWithContentsOfFile:(NSString *)path removingBlankSpaces:(BOOL)removeBlankSpaces {
    NSDictionary *dict = [self dictionaryWithContentsOfFile:path];
    if (removeBlankSpaces) {
        dict = [dict removeBlankSpaces];
    }
    return dict;
}

- (NSDictionary *)removeBlankSpaces {
    
    NSMutableDictionary *dict = [NSMutableDictionary dictionaryWithCapacity:self.count];
    for (NSString *key in self.allKeys) {
        id v = self[key];
        if ([v isKindOfClass:[NSString class]]) {
            dict[key] = [(NSString *)v removeBlankSpace];
        }
        else if ([v isKindOfClass:[NSDictionary class]]){
            dict[key] = [(NSDictionary *)v removeBlankSpaces];
        }
        else if ([v isKindOfClass:[NSArray class]]) {
            NSMutableArray *mv = [NSMutableArray array];
            for (id v1 in (NSArray *)v) {
                if ([v1 isKindOfClass:[NSString class]]) {
                    [mv addObject:[v1 removeBlankSpace]];
                }
                else if ([v1 isKindOfClass:[NSDictionary class]]){
                    [mv addObject:[(NSDictionary *)v1 removeBlankSpaces]];
                }
            }
            dict[key] = [NSArray arrayWithArray:mv];
        }
        else {
            dict[key] = v;
        }
    }
    
    
    return [NSDictionary dictionaryWithDictionary:dict];
}

@end

@implementation NSArray(EmptySpace)

- (NSArray *)removeBlankSpaces {
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:self.count];
    for (int i = 0; i < self.count; i++) {
        id obj = [self objectAtIndex:i];
        if (![obj isKindOfClass:[NSString class]]) {
            [array addObject:obj];
            continue;
        }
        
        if ([obj length] > 0) {
            [array addObject:obj];
        }
    }
    
    return [NSArray arrayWithArray:array];
}

@end
