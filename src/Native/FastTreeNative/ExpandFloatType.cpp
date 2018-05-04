//------------------------------------------------------------------------------
// <copyright company="Microsoft Corporation">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//------------------------------------------------------------------------------

#include "stdafx.h"
#define IsWeighted
#include "Sumup.h"
#include "SumupSegment.h"
#include "SumupNibbles.h"
#undef IsWeighted
#include "Sumup.h"
#include "SumupSegment.h"
#include "SumupNibbles.h"
#include "SumupOneBit.h"

// Ideally we should expand this using C++ templates. 
// However, In order to exporting functions from DLL float and double versions need to have different names (cannot be overloaded on type parameters)// Expanding here with ugly pre-processor macros to get double and float versions (with fucntion mes suffixes _float and _double)
// --andrzejp, 2010-03-05

#define FloatType float
#include "expand.h"
#undef FloatType

#define FloatType double
#include "expand.h"
#undef FloatType
