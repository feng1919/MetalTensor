//
//  AppDelegate.m
//  MobileNetV2
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#import "AppDelegate.h"
#import "ViewController.h"

@interface AppDelegate ()

@end

@implementation AppDelegate


- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions {
    // Override point for customization after application launch.
    
    self.window = [[UIWindow alloc] initWithFrame:[UIScreen mainScreen].bounds];
    self.window.rootViewController = [[ViewController alloc] init];
    
    [self.window makeKeyAndVisible];
    
    return YES;
}



@end
