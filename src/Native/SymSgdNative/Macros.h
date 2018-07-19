// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#define MAX(__X__, __Y__) (((__X__) < (__Y__)) ? (__Y__) : (__X__))
#define MIN(__X__, __Y__) (((__X__) > (__Y__)) ? (__Y__) : (__X__))

// This is a very large prime number used for permutation
#define VERYLARGEPRIME 961748941