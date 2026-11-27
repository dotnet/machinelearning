// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Link-time stand-in for the OpenMP runtime libiomp5.so. Do not install or package it.
// SymSgdNative links against it so its OpenMP imports name libiomp5.so with the VERSION
// symbol version, like libMklImports. At run time they resolve from the real libiomp5.so.
// If the compiler starts emitting a new OpenMP entry point, the link fails and it must be added here.

#include <stdio.h>
#include <stdlib.h>

__attribute__((constructor))
static void RejectRuntimeUse(void)
{
    fputs("The link-only libiomp5.so stub cannot be used as an OpenMP runtime.\n", stderr);
    abort();
}

#pragma GCC visibility push(default)

void __kmpc_barrier(void) {}
void __kmpc_end_reduce(void) {}
void __kmpc_for_static_fini(void) {}
void __kmpc_for_static_init_4(void) {}
void __kmpc_fork_call(void) {}
void __kmpc_global_thread_num(void) {}
void __kmpc_push_num_threads(void) {}
void __kmpc_reduce(void) {}
void omp_get_num_procs(void) {}
void omp_get_thread_num(void) {}
void omp_set_num_threads(void) {}

#pragma GCC visibility pop
