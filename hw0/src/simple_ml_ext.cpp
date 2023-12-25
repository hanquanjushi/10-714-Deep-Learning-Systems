#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
      int step = m/batch;
    for (int i = 0; i < step; i++) {
        int startidx = i*batch;
        int endidx = (i+1)*batch;
        float *X_batch = new float[batch*n];
        unsigned char *y_batch = new unsigned char[batch];
        for (int j = 0; j < batch; j++) {
            for (int k = 0; k < n; k++) {
                X_batch[j*n+k] = X[(startidx+j)*n+k];
            }
            y_batch[j] = y[startidx+j];
        }
        float *z = new float[batch*k];
        float *z_max = new float[batch];
        float *z_sum = new float[batch];
        float *gradient = new float[n*k];
        for (int j = 0; j < batch; j++) {
            z_max[j] = 0;
            for (int l = 0; l < k; l++) {
                z[j*k+l] = 0;
                for (int m = 0; m < n; m++) {
                    z[j*k+l] += X_batch[j*n+m] * theta[m*k+l];
                }
                if (z[j*k+l] > z_max[j]) {
                    z_max[j] = z[j*k+l];
                }
            }
        }
        for (int j = 0; j < batch; j++) {
            z_sum[j] = 0;
            for (int l = 0; l < k; l++) {
                z[j*k+l] = exp(z[j*k+l] - z_max[j]);
                z_sum[j] += z[j*k+l];
            }
        }
        for (int j = 0; j < batch; j++) {
            for (int l = 0; l < k; l++) {
                z[j*k+l] /= z_sum[j];
            }
        }
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                gradient[j*k+l] = 0;
                for (int m = 0; m < batch; m++) {
                    gradient[j*k+l] += X_batch[m*n+j] * (z[m*k+l] - (y_batch[m] == l));
                }
                gradient[j*k+l] /= batch;
            }
        }
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                theta[j*k+l] -= lr * gradient[j*k+l];
            }
        }
        delete[] X_batch;
        delete[] y_batch;
        delete[] z;
        delete[] z_max;
        delete[] z_sum;
        delete[] gradient;
        

    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
