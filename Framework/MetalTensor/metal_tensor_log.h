//
//  metal_tensor_log.h
//  MetalTensor
//
//  Created by Feng Stone on 2019/9/30.
//  Copyright © 2019 fengshi. All rights reserved.
//

#ifndef metal_tensor_log_h
#define metal_tensor_log_h

#include <stdio.h>

/* Control whether debugging macros are active at compile time */
#undef DB_ACTIVE
#ifdef DEBUG
#define DB_ACTIVE 1
#else
#define DB_ACTIVE 0
#endif /* DEBUG */

#define DB_TRACE(level, ...)\
            do { if (DB_ACTIVE) db_print(level, __VA_ARGS__); } while (0)

extern void     db_print(int level, const char *fmt, ...);

#endif /* metal_tensor_log_h */
