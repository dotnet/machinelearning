// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#pragma once
#include "../Stdafx.h"

enum ConfigParam
{
	/* Domain for forward transform. No default value */
	ForwardDomain = 0,

	/* Dimensionality, or rank. No default value */
	Dimension = 1,

	/* Length(s) of transform. No default value */
	Lengths = 2,

	/* Floating point precision. No default value */
	Precision = 3,

	/* Scale factor for forward transform [1.0] */
	ForwardScale = 4,

	/* Scale factor for backward transform [1.0] */
	BackwardScale = 5,

	/* Exponent sign for forward transform [Negative]  */
	/* ForwardSign = 6, ## NOT IMPLEMENTED */

	/* Number of data sets to be transformed [1] */
	NumberOfTransforms = 7,

	/* Storage of finite complex-valued sequences in complex domain
	   [ComplexComplex] */
	ComplexStorage = 8,

	/* Storage of finite real-valued sequences in real domain
	   [RealReal] */
	RealStorage = 9,

	/* Storage of finite complex-valued sequences in conjugate-even
	   domain [ComplexReal] */
	ConjugateEvenStorage = 10,

	/* Placement of result [InPlace] */
	Placement = 11,

	/* Generalized strides for input data layout [tigth, row-major for
	   C] */
	InputStrides = 12,

	/* Generalized strides for output data layout [tight, row-major
	   for C] */
	OutputStrides = 13,

	/* Distance between first input elements for multiple transforms
	   [0] */
	InputDistance = 14,

	/* Distance between first output elements for multiple transforms
	   [0] */
	OutputDistance = 15,

	/* Effort spent in initialization [Medium] */
	/* InitializationEffort = 16, ## NOT IMPLEMENTED */

	/* Use of workspace during computation [Allow] */
	/* Workspace = 17, ## NOT IMPLEMENTED */

	/* Ordering of the result [Ordered] */
	Ordering = 18,

	/* Possible transposition of result [None] */
	Transpose = 19,

	/* User-settable descriptor name [""] */
	DescriptorName = 20, /* DEPRECATED */

	/* Packing format for ComplexReal storage of finite
	   conjugate-even sequences [CcsFormat] */
	PackedFormat = 21,

	/* Commit status of the descriptor - R/O parameter */
	CommitStatus = 22,

	/* Version string for this DFTI implementation - R/O parameter */
	Version = 23,

	/* Ordering of the forward transform - R/O parameter */
	/* ForwardOrdering  = 24, ## NOT IMPLEMENTED */

	/* Ordering of the backward transform - R/O parameter */
	/* BackwardOrdering = 25, ## NOT IMPLEMENTED */

	/* Number of user threads that share the descriptor [1] */
	NumberOfUserThreads = 26
};

enum ConfigValue
{
	/* CommitStatus */
	Committed = 30,
	Uncommitted = 31,

	/* ForwardDomain */
	Complex = 32,
	Real = 33,
	/* ConjugateEven = 34,   ## NOT IMPLEMENTED */

	/* Precision */
	Single = 35,
	Double = 36,

	/* ForwardSign */
	/* Negative = 37,         ## NOT IMPLEMENTED */
	/* Positive = 38,         ## NOT IMPLEMENTED */

	/* ComplexStorage and ConjugateEvenStorage */
	ComplexComplex = 39,
	ComplexReal = 40,

	/* RealStorage */
	RealComplex = 41,
	RealReal = 42,

	/* Placement */
	InPlace = 43,          /* Result overwrites input */
	NotInPlace = 44,      /* Have another place for result */

	/* InitializationEffort */
	/* Low = 45,              ## NOT IMPLEMENTED */
	/* Medium = 46,           ## NOT IMPLEMENTED */
	/* High = 47,             ## NOT IMPLEMENTED */

	/* Ordering */
	Ordered = 48,
	BackwardScrambled = 49,
	/* ForwardScrambled = 50, ## NOT IMPLEMENTED */

	/* Allow/avoid certain usages */
	Allow = 51,            /* Allow transposition or workspace */
	/* Avoid = 52,            ## NOT IMPLEMENTED */
	None = 53,

	/* PackedFormat (for storing congugate-even finite sequence
	   in real array) */
	CcsFormat = 54,       /* Complex conjugate-symmetric */
	PackFormat = 55,      /* Pack format for real DFT */
	PermFormat = 56,      /* Perm format for real DFT */
	CceFormat = 57        /* Complex conjugate-even */
};

EXPORT_API(int) DftiSetValue(void *handle, ConfigParam config_param, ...);
EXPORT_API(int) DftiCreateDescriptor(void **handle, ConfigValue precision, ConfigValue domain, int dim, ...);
EXPORT_API(int) DftiCommitDescriptor(void *handle);
EXPORT_API(char *) DftiErrorMessage(int status);
EXPORT_API(int) DftiFreeDescriptor(void **handle);
EXPORT_API(int) DftiComputeForward(void *handle, ...);
EXPORT_API(int) DftiComputeBackward(void *handle, ...);

EXPORT_API(int) MKLDftiSetValue(void *handle, ConfigParam config_param, int config_val) {

	return DftiSetValue(handle, config_param, config_val);
}

EXPORT_API(int) MKLDftiCreateDescriptor(void **handle, ConfigValue precision, ConfigValue domain, int dim, int sizes)
{
	return DftiCreateDescriptor(handle, precision,domain, dim, sizes);
}

EXPORT_API(int) MKLDftiCommitDescriptor(void *handle)
{
	return DftiCommitDescriptor(handle);
}

EXPORT_API(char *) MKLDftiErrorMessage(int status)
{
	return DftiErrorMessage(status);
}

EXPORT_API(int) MKLDftiFreeDescriptor(void **handle)
{
	return DftiFreeDescriptor(handle);
}

EXPORT_API(int) MKLDftiComputeForward(void *handle, double *inputRe, double * inputIm, double * outputRe, double * outputIm) {

	return DftiComputeForward(handle, inputRe, inputIm, outputRe, outputIm);
}

EXPORT_API(int) MKLDftiComputeBackward(void *handle, double *inputRe, double * inputIm, double * outputRe, double * outputIm) {

	return DftiComputeBackward(handle, inputRe, inputIm, outputRe, outputIm);
}