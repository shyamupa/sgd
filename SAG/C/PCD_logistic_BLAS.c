#include <math.h>
#include "mex.h"
#include "blas.h"

/* 
 PCD_logistic(w,yX,lambda,L,jVals[,yXw]);
 *
 * Assumes that first column is bias variable
 * Last variable optionally provides the initial value yX*w to avoid doing this computation,
 * and this variable is updated in place.
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    int k,*jVals,maxIter,sparse;
    long i,j,nSamples,nVars,one=1;
    double gj,scaling,scaling2,*w,*yX,lambda,*L,*yXw,*tmp,*tmp2,*ones;
    char trans='N';
    mwIndex *jc, *ir;
    
    mxArray *rhs[1],*lhs[1];
    
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
            scaling = 1;
            scaling2 = 0;
            dgemv(&trans,&nSamples,&nVars,&scaling,yX,&nSamples,w,&one,&scaling2,yXw,&one);
        }
    }
    
    /* Container for input to Matlab's vector exp() function */
    rhs[0] = mxCreateDoubleMatrix(nSamples,1,mxREAL);
    /* We will replace the pointer in this container, but keep track
     * of the original memory for later de-allocation */
    tmp2 = mxGetPr(rhs[0]);
        
    /* Vector of ones for computing sums with ddot */
    ones = mxCalloc(nSamples,sizeof(double));
    for(i = 0;i < nSamples; i++)
        ones[i] = 1;
    
    for(k=0;k<maxIter;k++)
    {
        /* Select next variable to update */
        j = jVals[k]-1;
        
        gj = 0;
        if (sparse && j != 0) {
            for(i = jc[j];i < jc[j+1];i++) {
                gj += yX[i]/(1+exp(yXw[ir[i]]));
            }
        }
        else {
            /* Call Matlab's 'exp' function (faster than looping) */
            mxSetPr(rhs[0],yXw);
            mexCallMATLAB(1,lhs,1,rhs,"exp");
            tmp = mxGetPr(lhs[0]);
                        
            for(i = 0;i < nSamples; i++)
                tmp[i] = yX[i + nSamples*j]/(1+tmp[i]);
                     
            gj = ddot(&nSamples,tmp,&one,ones,&one);
            mxDestroyArray(lhs[0]);
        }
        gj = -gj/nSamples + w[j]*lambda;
        
        w[j] -= gj/L[j];
        
        if (sparse && j != 0) {
            for(i = jc[j];i < jc[j+1];i++) {
                yXw[ir[i]] -= yX[i]*gj/L[j];
            }
        }
        else {
            scaling = -gj/L[j];
            daxpy(&nSamples,&scaling,&yX[nSamples*j],&one,yXw,&one);
        }
    }
    
    if (nrhs != 6)
        mxFree(yXw);
    mxFree(ones);
    mxSetPr(rhs[0],tmp2);
    mxDestroyArray(rhs[0]);
    
}
