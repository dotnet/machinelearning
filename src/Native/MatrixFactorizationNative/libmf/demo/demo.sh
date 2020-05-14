#!/bin/sh
train=../mf-train
predict=../mf-predict

##########################################################################
# Build package if no binary found and this script is exectuted via the
# following command.
#  libmf/demo > sh demo.sh
##########################################################################
if [ ! -s $train ] || [ ! -s $predict ]
then
    (cd .. && make)
fi

##########################################################################
# Real-valued matrix factorization (RVMF)
##########################################################################
echo "--------------------------------"
echo "Real-valued matrix factorization"
echo "--------------------------------"
# In-memory training with holdout valudation
$train -f 0 -l2 0.05 -k 100 -t 10 -p real_matrix.te.txt real_matrix.tr.txt rvmf_model.txt
# Do prediction and show MAE
$predict -e 1 real_matrix.te.txt rvmf_model.txt rvmf_output.txt

##########################################################################
# Binary matrix factorization (BMF)
##########################################################################
echo "---------------------------"
echo "binary matrix factorization"
echo "---------------------------"
# In-memory training with holdout valudation
$train -f 5 -l2 0.01 -k 64 -p binary_matrix.te.txt binary_matrix.tr.txt bmf_model.txt
# Do prediction and show accuracy
$predict -e 6 binary_matrix.te.txt bmf_model.txt bmf_output.txt

##########################################################################
# One-class matrix factorization (OCMF)
##########################################################################
echo "-----------------------------------------------------------------"
echo "one-class matrix factorization using a stochastic gradient method"
echo "-----------------------------------------------------------------"
# In-memory training with holdout validation
$train -f 10 -l2 0.01 -k 32 -p all_one_matrix.te.txt all_one_matrix.tr.txt ocmf_model.txt
# Do prediction and show row-oriented MPR
$predict -e 10 all_one_matrix.te.txt ocmf_model.txt ocmf_output.txt
# Do prediction and show row-oriented AUC
$predict -e 12 all_one_matrix.te.txt ocmf_model.txt ocmf_output.txt

echo "----------------------------------------------------------------"
echo "one-class matrix factorization using a coordinate descent method"
echo "----------------------------------------------------------------"
# In-memory training with holdout validation
$train -f 12 -l2 0.01 -k 32 -a 0.001 -c 0.0001 -p all_one_matrix.te.txt all_one_matrix.tr.txt ocmf_model.txt
# Do prediction and show row-oriented MPR
$predict -e 10 all_one_matrix.te.txt ocmf_model.txt ocmf_output.txt
# Do prediction and show row-oriented AUC
$predict -e 12 all_one_matrix.te.txt ocmf_model.txt ocmf_output.txt
