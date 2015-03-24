// -*- C++ -*-
// SVM with stochastic gradient
// Copyright (C) 2007- Leon Bottou

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111, USA

#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <cstdlib>
#include <cmath>

#include "assert.h"
#include "vectors.h"
#include "gzstream.h"
#include "timer.h"
#include "loss.h"
#include "data.h"

#if HAS_UNIFORMINTDISTRIBUTION
# include <random>
typedef std::uniform_int_distribution<int> uniform_int_generator;
#elif HAS_UNIFORMINT
# include <tr1/random>
typedef std::tr1::uniform_int<int> uniform_int_generator;
#else
struct uniform_int_generator { 
  int imin, imax; 
  uniform_int_generator(int imin, int imax) : imin(imin),imax(imax) {}
  int operator()() { return imin + std::rand() % (imax - imin + 1); }
};
#endif

using namespace std;

// ---- Loss function

// Compile with -DLOSS=xxxx to define the loss function.
// Loss functions are defined in file loss.h)
#ifndef LOSS
# define LOSS LogLoss
#endif

// ---- Bias term

// Compile with -DBIAS=[1/0] to enable/disable the bias term.
// Compile with -DREGULARIZED_BIAS=1 to enable regularization on the bias term

#ifndef BIAS
# define BIAS 1
#endif
#ifndef REGULARIZED_BIAS
# define REGULARIZED_BIAS 0
#endif

// ---- Plain stochastic gradient descent

class SvmSvrg
{
public:
  SvmSvrg(int dim, double lambda);
  void init(int imin,int imax);
  void renorm();
  double wnorm();
  double testOne(const SVector &x, double y, double *ploss, double *pnerr);
  void trainOne(const SVector &x, double y,double eta,int i);
  void computeMu(int imin,int imax,const xvec_t &xp, const yvec_t &yp);
public:
  void train(int imin, int imax,int m, double eta, const xvec_t &x, const yvec_t &y, const char *prefix = "");
  void test(int imin, int imax, const xvec_t &x, const yvec_t &y, const char *prefix = "");
public:
  double evaluateEta(int imin, int imax, const xvec_t &x, const yvec_t &y, double eta);
  const char* dataFile;
	
private:
  double  lambda;
  FVector w;
  FVector wt;			// w tilde in svrg paper
  FVector mu;			// mu from same paper
  FVector saved_d;		// saved_d wrt wt
  int saved_dimin;
  int saved_dimax;
  double  wBias;
  double  wtBias;
  double  t;
  int numGradEvals;
};

/// Constructor
SvmSvrg::SvmSvrg(int dim, double lambda)
  : lambda(lambda), 
    w(dim), wt(dim), mu(dim),
    saved_dimin(0), saved_dimax(-1), wBias(0),wtBias(0),
    t(0), numGradEvals(0), dataFile("svrg.dat")
{
}

/// Compute the norm of the weights
double
SvmSvrg::wnorm()
{
  double norm = dot(w,w);
#if REGULARIZED_BIAS
  norm += wBias * wBias;
#endif
  return norm;
}

/// Compute the output for one example.
double
SvmSvrg::testOne(const SVector &x, double y, double *ploss, double *pnerr)
{
  double s = dot(w,x) + wBias;
  if (ploss)
    *ploss += LOSS::loss(s, y);
  if (pnerr)
    *pnerr += (s * y <= 0) ? 1 : 0;
  return s;
}

/// Perform one iteration of the SGD algorithm with specified gains
void
SvmSvrg::trainOne(const SVector &x, double y, double eta, int i)
{
  double s = dot(w,x)  + wBias;
  // update for regularization term
  // update for loss term
  double d = LOSS::dloss(s, y);
  double dt;
  if (d != 0)
    {
      w.scale(1-eta*lambda);	// w = (1-eta*lambda)w
      w.add(x, eta * d); 	// usual
      dt=saved_d[i-saved_dimin]; // fetch the loss computed by wt
      w.add(x, - eta * dt);	// first correction term
      w.add(mu,eta);		// mu term
    }
  numGradEvals++;
  // same for the bias
#if BIAS
  double etab = eta * 0.01;
#if REGULARIZED_BIAS
  wBias *= (1 - etab * lambda);
#endif
  wBias += etab * d;
#endif
}

/// computes 1/n * sum_i dL(wTx_i),y_i)/dw
void
SvmSvrg::computeMu(int imin,int imax,const xvec_t &xp, const yvec_t &yp)
{
  mu=FVector::FVector(mu.size()); // reset
  double s,d;
  for (int i=imin; i<=imax; i++)
    {
      SVector x=xp.at(i);
      double y=yp.at(i);
      s=dot(wt,x) + wtBias;
      d=LOSS::dloss(s,y);
      mu.add(x,d);
      saved_d[i-saved_dimin]=d;
      numGradEvals++;
    }
  mu.scale(1.0/(imax-imin+1));
}
/// Perform a training epoch
void 
SvmSvrg::train(int imin, int imax, int m, double eta, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Training on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  // assert(eta0 > 0);

  // compute Mu, also save f'(wt) for all i
  computeMu(imin,imax,xp,yp);
  w=wt.slice(0,wt.size()-1);	// w_0 = w_t
  cout << "norm " << wnorm() << endl;
  cout << "decay " << (1-eta*lambda) << endl;
  uniform_int_generator generator(imin, imax);
  for (int i=0; i<m; i++)
    {
      int ii = generator(); 
      trainOne(xp.at(ii), yp.at(ii), eta,ii);
    }
  wt=w.slice(0,w.size()-1);	// option I
  cout << prefix << setprecision(6) << "wNorm=" << wnorm();
#if BIAS
  cout << " wBias=" << wBias;
#endif
  cout << endl;
  // Writing to file                                                            
  FILE *f = fopen(dataFile, "a");
  fprintf(f, "\n%lf", (numGradEvals*1.0/(imax-imin+1)));
  fclose(f);
}

void
SvmSvrg::init(int imin,int imax)
{
  saved_d.resize(imax - imin + 1);
  saved_dimin = imin;
  saved_dimax = imax;
}

/// Perform a test pass
void 
SvmSvrg::test(int imin, int imax, const xvec_t &xp, const yvec_t &yp, const char *prefix)
{
  cout << prefix << "Testing on [" << imin << ", " << imax << "]." << endl;
  assert(imin <= imax);
  double nerr = 0;
  double loss = 0;
  for (int i=imin; i<=imax; i++)
    testOne(xp.at(i), yp.at(i), &loss, &nerr);
  nerr = nerr / (imax - imin + 1);
  loss = loss / (imax - imin + 1);
  double cost = loss + 0.5 * lambda * wnorm();
  cout << prefix 
       << "Loss=" << setprecision(12) << loss
       << " Cost=" << setprecision(12) << cost 
       << " Misclassification=" << setprecision(4) << 100 * nerr << "%." 
       << endl;
  // Writing to file                                                            
  FILE *f = fopen(dataFile, "a");
  fprintf(f, "\t%lf\t%lf\t%lf", loss, cost, (100*nerr));
  fclose(f);
}

/// Perform one epoch with fixed eta and return cost

double 
SvmSvrg::evaluateEta(int imin, int imax, const xvec_t &xp, const yvec_t &yp, double eta)
{
  SvmSvrg clone(*this); // take a copy of the current state
  assert(imin <= imax);
  for (int i=imin; i<=imax; i++)
    clone.trainOne(xp.at(i), yp.at(i), eta,i);
  double loss = 0;
  double cost = 0;
  for (int i=imin; i<=imax; i++)
    clone.testOne(xp.at(i), yp.at(i), &loss, 0);
  loss = loss / (imax - imin + 1);
  cost = loss + 0.5 * lambda * clone.wnorm();
  // cout << "Trying eta=" << eta << " yields cost " << cost << endl;
  return cost;
}


// --- Command line arguments

const char *trainfile = 0;
const char *testfile = 0;
bool normalize = true;
double lambda = 1e-5;
int epochs = 5;
int maxtrain = -1;
double eta=0.025;

void
usage(const char *progname)
{
  const char *s = ::strchr(progname,'/');
  progname = (s) ? s + 1 : progname;
  cerr << "Usage: " << progname << " [options] trainfile [testfile]" << endl
       << "Options:" << endl;
#define NAM(n) "    " << setw(16) << left << n << setw(0) << ": "
#define DEF(v) " (default: " << v << ".)"
  cerr << NAM("-lambda x")
       << "Regularization parameter" << DEF(lambda) << endl
       << NAM("-epochs n")
       << "Number of training epochs" << DEF(epochs) << endl
       << NAM("-eta e")
       << "Constant step length" << DEF(eta) << endl
       << NAM("-dontnormalize")
       << "Do not normalize the L2 norm of patterns." << endl
       << NAM("-maxtrain n")
       << "Restrict training set to n examples." << endl;
#undef NAM
#undef DEF
  ::exit(10);
}

void
parse(int argc, const char **argv)
{
  for (int i=1; i<argc; i++)
    {
      const char *arg = argv[i];
      if (arg[0] != '-')
        {
          if (trainfile == 0)
            trainfile = arg;
          else if (testfile == 0)
            testfile = arg;
          else
            usage(argv[0]);
        }
      else
        {
          while (arg[0] == '-') 
            arg += 1;
          string opt = arg;
          if (opt == "lambda" && i+1<argc)
            {
              lambda = atof(argv[++i]);
              assert(lambda>0 && lambda<1e4);
            }
          else if (opt == "epochs" && i+1<argc)
            {
              epochs = atoi(argv[++i]);
              assert(epochs>0 && epochs<1e6);
            }
	  else if (opt == "eta" && i+1<argc)
            {
              eta = atof(argv[++i]);
              assert(eta>0 && eta*lambda<1);
            }
          else if (opt == "dontnormalize")
            {
              normalize = false;
            }
          else if (opt == "maxtrain" && i+1 < argc)
            {
              maxtrain = atoi(argv[++i]);
              assert(maxtrain > 0);
            }
          else
            {
              cerr << "Option " << argv[i] << " not recognized." << endl;
              usage(argv[0]);
            }

        }
    }
  if (! trainfile)
    usage(argv[0]);
}

void 
config(const char *progname)
{
  cout << "# Running: " << progname;
  cout << " -lambda " << lambda;
  cout << " -epochs " << epochs;
  cout << " -eta " << eta;
  if (! normalize) cout << " -dontnormalize";
  if (maxtrain > 0) cout << " -maxtrain " << maxtrain;
  cout << endl;
#define NAME(x) #x
#define NAME2(x) NAME(x)
  cout << "# Compiled with: "
       << " -DLOSS=" << NAME2(LOSS)
       << " -DBIAS=" << BIAS
       << " -DREGULARIZED_BIAS=" << REGULARIZED_BIAS
       << endl;
}

// --- main function

int dims;
xvec_t xtrain;
yvec_t ytrain;
xvec_t xtest;
yvec_t ytest;

int main(int argc, const char **argv)
{
  parse(argc, argv);
  config(argv[0]);
  if (trainfile)
    load_datafile(trainfile, xtrain, ytrain, dims, normalize, maxtrain);
  if (testfile)
    load_datafile(testfile, xtest, ytest, dims, normalize);
  cout << "# Number of features " << dims << "." << endl;
  // prepare svm
  int imin = 0;
  int imax = xtrain.size() - 1;
  int tmin = 0;
  int tmax = xtest.size() - 1;
  SvmSvrg svm(dims, lambda);

  svm.init(imin,imax);
  Timer timer;
  // determine eta0 using sample
  // int smin = 0;
  // int smax = imin + min(1000, imax);
  // timer.start();
  // svm.determineEta0(smin, smax, xtrain, ytrain);
  // timer.stop();

  // train
  int m= (imax-imin+1) / 10;
  cout << "Using update freq m = "<< m << endl;

  // Writing to file                                                            
  FILE *f = fopen(svm.dataFile, "w");
  fprintf(f, "numGradEvals\tTrainLoss\tTrainCost\tTrainErr\tTestLoss\tTestCost\tTestErr");
  fclose(f);

  for(int i=0; i<epochs; i++)
    {
      cout << "--------- Epoch " << i+1 << "." << endl;
      timer.start();
      svm.train(imin, imax,m,eta, xtrain, ytrain);
      timer.stop();
      cout << "Total training time " << setprecision(6) 
           << timer.elapsed() << " secs." << endl;
      svm.test(imin, imax, xtrain, ytrain, "train: ");
      if (tmax >= tmin)
        svm.test(tmin, tmax, xtest, ytest, "test:  ");
    }
  return 0;
}
