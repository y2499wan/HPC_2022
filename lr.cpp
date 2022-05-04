#include <cmath>
#include <memory>
#include <iostream>
#include <random>
#include "utility.h"
#include "lr.h"
#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

double dot_product(double* a, double* b, int n) { 
    double out = 0.0;
    for (long i = 0; i < n; i++) {
        out += a[i] * b[i];
    }
    return out;
}

LogisticRegression::LogisticRegression(int features, int nthreads)
{
    new_weights = new double[features];
    old_weights = new double[features];
    total_l1 = new double[features];
    features_num = features;
    pold_weights = new_2d_mat(nthreads, features_num);
    pnew_weights = new_2d_mat(nthreads, features_num);
    bias = 0.0;
    bias_old = 0.0;
}

double LogisticRegression::sigmoid(double x){

    static double overflow = 20.0;
    if (x > overflow) x = overflow;
    if (x < -overflow) x = -overflow;
    return 1.0/(1.0 + exp(-x));
}

// generate random number between 0 and 1
double LogisticRegression::rand_generator() {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0,1);
    return distr(eng);
}

// l2 norm between array a and b
double LogisticRegression::l2norm (double* a, double* b, int n) {
    double out = 0.0;
    for (long i = 0; i < n; i++) {
        out += pow(a[i] - b[i],2);
    }
    return sqrt(out);
}

double LogisticRegression::classify(double* x) {
    return classify_wrapper(x, new_weights, features_num, bias);
}

double LogisticRegression::classify_wrapper(double* x, double* new_weights, int features_num, double bias) {
    double logit = dot_product(x, new_weights, features_num) + bias;
    return sigmoid(logit);
}

// Parallel SGD: put average sgd in new weights
void LogisticRegression::averageWeights(int nthreads) {
    memset(new_weights, 0, sizeof(double) * features_num);
    // #pragma omp parallel for collapse(2)//num_threads(nthreads)
    for (int i = 0; i < features_num; i++) {
        for (int j = 0; j < nthreads; j++) {
            //#pragma omp atomic update
            new_weights[i] += pnew_weights[j][i]/nthreads;
        }
    }
}

// training data(x) and class i(y)
double LogisticRegression::fit(double **x, double *y, int m, int n, double alpha, double l1, int max_iter) {
    
    memset(old_weights, 0, sizeof(double) * features_num);
    memset(new_weights, 0, sizeof(double) * features_num);
    memset(total_l1, 0, sizeof(double) * features_num);

    // initialize weights
    //for (int i = 0; i < features_num; i++) old_weights[i] = rand_generator();

    double *predict = new double[m];
    double mu = 0.0;
    double norm = 1.0;
    double loss;

    for (int iter = 0; iter<max_iter; iter++) {
        // cout << "iter " << iter << endl;
        loss = 0.0;
        for (int i=0; i<m; i++) {
            predict[i] = classify_wrapper(x[i], old_weights, features_num, bias_old); //sigmoid
            for (int j=0; j<features_num; j++) {
                double gradient = (predict[i] - y[i])*x[i][j];
                new_weights[j] = old_weights[j] - alpha*gradient - l1 * old_weights[j];
            }
            loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
            std::swap(old_weights, new_weights);
        }
        //cout << "cross_entropy_loss:" << loss << endl;     
    }
    delete [] predict;
    return loss;
}

double LogisticRegression::sgd(double **x, double *y, int m, int n, double alpha, double l1, int max_iter, int nthreads, int T, int tid) {
    double norm = 1.0;
    int iter = 0;
    double loss = 0;
    double *predict = new double[m];
    for (int i=tid*T; i< tid*T+T; i++) {
        predict[i] = classify_wrapper(x[i], pold_weights[tid], features_num, bias_old);
        for (int j=0; j<features_num; j++) {
            double gradient = (predict[i] - y[i])*x[i][j];
            pnew_weights[tid][j] = pold_weights[tid][j] - alpha*gradient - l1 * pold_weights[tid][j];
        }
        loss += -((y[i] * log(predict[i]) + (1 - y[i]) * log(1 - predict[i])) / m);
        std::swap(pold_weights[tid], pnew_weights[tid]);
    }
    delete [] predict;
    return loss;
}

double LogisticRegression::parallelFit(double **x, double *y, int m, int n, double alpha, double l1, int max_iter, int nthreads) {
  
    // #pragma omp parallel for num_threads(nthreads)
    // for (int j = 0; j < nthreads; j++) {
    //     for (int i = 0; i < features_num; i++) {
    //         pold_weights[j][i] = rand_generator();
    //     }
    // }
    double *predict = new double[m];
    int tid;
    int T = (int) m/nthreads; // each thread do T rows
    double loss; 

    for (int iter = 0; iter<max_iter; iter++) {
        loss = 0.0;
        #pragma omp parallel for num_threads(nthreads) default(shared) private(tid) 
        for (int i=0; i < nthreads; i++) {
            tid = omp_get_thread_num();
            #pragma omp atomic update
            loss += sgd(x, y, m, n, alpha, l1, max_iter, nthreads, T, tid);
        }
        #pragma omp barrier
    }
    // average over weights 
    averageWeights(nthreads);
    return loss;
}


