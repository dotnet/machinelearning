/*
Copyright(c) 2014 - 2015 The LIBMF Project.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met :

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and / or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES(INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT(INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef _LIBMF_H
#define _LIBMF_H

#ifdef __cplusplus
extern "C" 
{

namespace mf
{
#endif

// Changing the following typedef is not allowed in this version.
typedef float mf_float;
typedef double mf_double;
typedef int mf_int;
typedef long long mf_long;

struct mf_node
{
    mf_int u;
    mf_int v;
    mf_float r;
};

struct mf_problem
{
    mf_int m;
    mf_int n;
    mf_long nnz;
    struct mf_node *R;
};

struct mf_parameter
{
    mf_int k; 
    mf_int nr_threads;
    mf_int nr_bins;
    mf_int nr_iters;
    mf_float lambda; 
    mf_float eta;
    mf_int do_nmf;
    mf_int quiet; 
    mf_int copy_data;
};

struct mf_parameter mf_get_default_param();

struct mf_model
{
    mf_int m;
    mf_int n;
    mf_int k;
    mf_float *P;
    mf_float *Q;
};

mf_int mf_save_model(struct mf_model const *model, char const *path);

struct mf_model* mf_load_model(char const *path);

void mf_destroy_model(struct mf_model **model);

struct mf_model* mf_train(
    struct mf_problem const *prob, 
    struct mf_parameter param);

struct mf_model* mf_train_with_validation(
    struct mf_problem const *tr, 
    struct mf_problem const *va, 
    struct mf_parameter param);

mf_float mf_cross_validation(
    struct mf_problem const *prob, 
    mf_int nr_folds, 
    struct mf_parameter param);

mf_float mf_predict(struct mf_model const *model, mf_int p_idx, mf_int q_idx);

#ifdef __cplusplus
} // namespace mf

} // extern "C"
#endif

#endif // _LIBMF_H
