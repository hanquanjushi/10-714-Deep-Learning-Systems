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
    // Allocate memory for logits and gradients arrays
    float *logits = new float[m * k]();
    float *gradients = new float[n * k]();

    // Loop over minibatches
    for (size_t start = 0; start < m; start += batch) {
        size_t end = std::min(start + batch, m);

        // Calculate logits for each example
        for (size_t i = start; i < end; i++) {
            for (size_t j = 0; j < k; j++) {
                for (size_t l = 0; l < n; l++) {
                    logits[i * k + j] += X[i * n + l] * theta[l * k + j];
                }
            }
        }

        // Apply softmax function to logits
        for (size_t i = start; i < end; i++) {
            float max_logit = logits[i * k];
            for (size_t j = 1; j < k; j++) {
                max_logit = std::max(max_logit, logits[i * k + j]);
            }

            float sum_exp_logit = 0.0;
            for (size_t j = 0; j < k; j++) {
                logits[i * k + j] = std::exp(logits[i * k + j] - max_logit);
                sum_exp_logit += logits[i * k + j];
            }

            for (size_t j = 0; j < k; j++) {
                logits[i * k + j] /= sum_exp_logit;
            }
        }

        // Calculate gradients and accumulate over minibatch
        for (size_t i = start; i < end; i++) {
            for (size_t j = 0; j < k; j++) {
                gradients[j] += (logits[i * k + j] - (y[i] == j)) * X[i * n];
            }

            for (size_t l = 1; l < n; l++) {
                for (size_t j = 0; j < k; j++) {
                    gradients[l * k + j] += (logits[i * k + j] - (y[i] == j)) * X[i * n + l];
                }
            }
        }

        // Update theta values
        for (size_t l = 0; l < n; l++) {
            for (size_t j = 0; j < k; j++) {
                theta[l * k + j] -= (lr / batch) * gradients[l * k + j];
            }
        }

        // Reset gradients for next minibatch
        std::fill(gradients, gradients + n * k, 0.0);
    }

    // Deallocate memory for logits and gradients arrays
    delete[] logits;
    delete[] gradients;
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
