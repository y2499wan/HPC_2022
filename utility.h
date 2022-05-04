using namespace std;
#include <string>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iostream>

#ifndef _UTILS_H_
#define _UTILS_H_
class Timer {
  public:

    void tic() {
      t_start = std::chrono::high_resolution_clock::now();
    }

    double toc() {
      auto t_end = std::chrono::high_resolution_clock::now();
      return std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count() * 1e-9;
    }

  private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
};

#endif

extern double** new_2d_mat(int row, int col);
extern void load_feature_matrix(const char* train_feature, double**x);
extern double* double_array(int row);
extern double **normalization(double ** x,int row,int col);
extern void split(int train_row, int train_col, double** train_data, double** train_x,double* train_y);
//extern void predict(vector<LogisticRegression>n models, int train_col, int train_row, int num_classes, double* train_y,
//double** normalized_x, double** pred_train, int& train_correct);