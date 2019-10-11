//
//  metal_tensor_log.c
//  MetalTensor
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#include "metal_tensor_log.h"
#include <stdarg.h>

static int      debug = 0;

/*
**  Call as: db_print(level, format, ...);
**  Print debug information if debug flag set at or above level.
*/
void db_print(int level, const char *fmt, ...)
{
    if (debug >= level)
    {
        va_list args;
        va_start(args, fmt);
        printf(fmt, args);
        va_end(args);
    }
}

