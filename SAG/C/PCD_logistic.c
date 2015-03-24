#include <math.h>
#include "mex.h"

/* 
 PCD_logistic(w,yX,lambda,L,jVals[,yXw]);
 *
 * Assumes that first column is bias variable
 * Last variable optionally provides the initial value yX*w to avoid doing this computation,
 * and this variable is updated in place.
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    int k,*jVals,nVars,maxIter,sparse;
    long i,j,nSamples;
    double gj,*w,*yX,lambda,*L,*yXw;
    mwIndex *jc, *ir;
    
    if (nrhs < 5)
        mexErrMsgTxt("At least 5 arguments are needed: {w,yX,lambda,L,jVals[,yXw]}");
    
    w = mxGetPr(prhs[0]);
    yX = mxGetPr(prhs[1]);
    lambda = mxGetScalar(prhs[2]);
    L = mxGetPr(prhs[3]);
    jVals = (int*)mxGetPr(prhs[4]);  
    
    if (!mxIsClass(prhs[4],"int32"))
        mexErrMsgTxt("jVals must be int32");
    
    /* Compute Sizes */
    nSamples = mxGetM(prhs[1]);
    nVars = mxGetN(prhs[1]);
    maxIter = mxGetM(prhs[4]);
    
    /* Basic input checking */
    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("the number or rows in w should be the same as the number of columns of yX");
    if (1 != mxGetN(prhs[0]))
        mexErrMsgTxt("w should only have one column");
    if (nVars != mxGetM(prhs[3]))
        mexErrMsgTxt("the number of rows should be the same as the number of columns of yX");
    if (1 != mxGetN(prhs[3]))
        mexErrMsgTxt("L should only have one row");
    
    sparse = 0;
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
        
    /* Initialize yXw */
    if (nrhs ==6)
    {
        yXw = mxGetPr(prhs[5]);
        if (nSamples != mxGetM(prhs[5]))
            mexErrMsgTxt("the number of rows in yXw should be the same as the number of rows in yX");
        if (1 != mxGetN(prhs[5]))
            mexErrMsgTxt("yXw should only have one column");
    }
    else
    {
        yXw = mxCalloc(nSamples,sizeof(double));
        if (sparse) {
            for(j = 0;j < nVars;j++) {
                for(i = jc[j];i < jc[j+1];i++)
                    yXw[ir[i]] += yX[i]*w[j];
            }
        }
        else {
            for(j = 0;j < nVars;j++) {
                for(i= 0;i < nSamples;i++) {
                    yXw[i] += yX[i + nSamples*j]*w[j];
                }
            }
        }
    }
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next variable to update */
        j = jVals[k]-1;
        
        gj = 0;
        if (j == 0) { /* Bias variable, not sparse */
            for(i = 0;i < nSamples; i++) {
                gj += yX[i]/(1+exp(yXw[i]));
            }
        }
        else if (sparse) {
            for(i = jc[j];i < jc[j+1];i++) {
                gj += yX[i]/(1+exp(yXw[ir[i]]));
            }
        }
        else {
            for(i = 0;i < nSamples; i++) {
                gj += yX[i + nSamples*j]/(1+exp(yXw[i]));
            }
        }
        gj = -gj/nSamples + w[j]*lambda;
        
        w[j] -= gj/L[j];
        
        if (j == 0) {
            for(i = 0;i < nSamples;i++) {
                yXw[i] -= yX[i]*gj/L[j];
            }
        }
        else if (sparse) {
            for(i = jc[j];i < jc[j+1];i++) {
                yXw[ir[i]] -= yX[i]*gj/L[j];
            }
        }
        else {
            for(i = 0;i < nSamples;i++) {
                yXw[i] -= yX[i + nSamples*j]*gj/L[j];
            }
        }

    }
    
    if (nrhs != 6)
        mxFree(yXw);
    
}
