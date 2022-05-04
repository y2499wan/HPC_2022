#include<cstring>
#include <fstream>
#include "utility.h"
using namespace std;

class LogisticRegression
{
private:
    double *old_weights;
    double *new_weights;
    double *total_l1;
    int features_num;
    double bias;
    double bias_old;
    double **pold_weights;
    double **pnew_weights;
public:
    LogisticRegression(int features, int nthreads);
    double sigmoid(double);
    double rand_generator();
    double l2norm (double*, double*, int);
    double classify(double*);
    double fit(double**, double*, int, int, double, double, int);
    double classify_wrapper(double*, double*, int, double);
    double parallelFit(double **, double *, int, int, double, double, int, int);
    double sgd(double **, double *, int, int, double, double, int, int, int, int);
    void averageWeights(int);
};

