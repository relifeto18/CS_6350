#!/bin/sh

echo SVM in the primal domain
python3 SVM_primal.py

echo

echo SVM in the dual domain
python3 SVM_dual.py

echo

echo Dual SVM with a Gaussian kernel
python3 Gaussian_kernel.py
