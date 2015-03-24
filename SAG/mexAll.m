fprintf('Compiling normal C files...\n');
mex C/sampleDiscrete_cumsumC.c -largeArrayDims
mex C/SGD_logistic.c -largeArrayDims
mex C/ASGD_logistic.c -largeArrayDims
mex C/PCD_logistic.c -largeArrayDims
mex C/DCA_logistic.c -largeArrayDims
mex C/SAG_logistic.c -largeArrayDims
mex C/SAGlineSearch_logistic.c -largeArrayDims
mex C/SAG_LipschitzLS_logistic.c -largeArrayDims

fprintf('Compiling BLAS C files...\n');
mex C/SGD_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/ASGD_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/PCD_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/DCA_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/SAG_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/SAGlineSearch_logistic_BLAS.c -largeArrayDims -lmwblas
mex C/SAG_LipschitzLS_logistic_BLAS.c -largeArrayDims -lmwblas
