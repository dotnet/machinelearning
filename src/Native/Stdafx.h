// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include <limits>
#include <assert.h>
#include <cmath>
#include <cstring>

#define UNUSED(x) (void)(x)
#define DEBUG_ONLY(x) (void)(x)

#ifdef _WIN32
#include <intrin.h>

#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret
#else
#include "UnixSal.h"

#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret

#define __forceinline __attribute__((always_inline)) inline
#endif