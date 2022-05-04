#include<cstring>
#include <fstream>
#include "utility.h"
#include "lr.h"
#include <cmath>
using namespace std;


double** new_2d_mat(int row, int col) {
    double** x = new double*[row];
    for(int i = 0; i < row; ++i) {
        x[i] = new double[col];
        memset(x[i], 0, sizeof(double)*col);
    }
    return x;
}

void csv_read(const char* in_string, double*x) {
    const char* pos = in_string;
    int i = 0;
    for(; pos - 1 != NULL && pos[0] != '\0' && pos[0] != '#';)
    {
        float value = atof(pos);
        x[i++] = value;
        pos = strchr(pos, ',') + 1;
    }
}

void load_feature_matrix(const char* train_feature, double** x) {
    int cur_row = 0;
    ifstream ifs(train_feature);
    string line;
    while(getline(ifs, line)) {
        csv_read(line.c_str(), x[cur_row]);
        cur_row++;
    }
}

double* double_array(int row) {
    double* x = new double[row];
    memset(x, 0, sizeof(double)*row);
    return x;
}

double **normalization(double** x, int m, int n)
{
    double **scale_x = new_2d_mat(m, n);
    for (int i = 0; i < n; ++i)
    { 
        double mean = 0.0;
        double var = 0.0;
        for (int j = 0; j < m; j++)
        {
            mean += x[j][i];
        }
        mean = mean / m;
        for (int j = 0; j < m; j++)
        {
            var += pow(x[j][i] - mean,2); // (x_i-mean)^2/(n-1)
        }
        var = var / (m-1);
        double std = sqrt(var);
        for (int j = 0; j < m; j++)
        {
            scale_x[j][i] = (x[j][i] - mean) / std;
        }
    }
    return scale_x;
}

void split(int train_row, int train_col, double** train_data, double** train_x,double* train_y) {
    for (long i = 0; i < train_row; i++)
    {
        for (long j = 0; j < train_col+1; j++)
        {
            if (j < train_col)
                train_x[i][j] = train_data[i][j];
            else
                train_y[i] = int(train_data[i][j]);
        }
    }

}
