//
//  NSString+Extension.h
//  MetalImage
//
//  Created by Feng Stone on 2019/6/25.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NSString (Extension)

- (NSString *)removeBlankSpace;
- (NSArray *)nonEmptyComponentsSeparatedByString:(NSString *)string;

@end

@interface NSDictionary (blankSpaces)

+ (NSDictionary *)dictionaryWithContentsOfFile:(NSString *)path removingBlankSpaces:(BOOL)removeBlankSpaces;
- (NSDictionary *)removeBlankSpaces;

@end

@interface NSArray (EmptySpace)

- (NSArray *)removeBlankSpaces;

@end

NS_ASSUME_NONNULL_END
