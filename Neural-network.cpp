#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;

MatrixXd sigmoid(const MatrixXd &z) {
    return 1.0 / (1.0 + (-z.array()).exp());
}

MatrixXd sigmoid_derivative(const MatrixXd &z) {
    return sigmoid(z).array() * (1 - sigmoid(z).array());
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> initialize_parameters(int input_size, int hidden_size, int output_size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    MatrixXd W1(hidden_size, input_size);
    W1 = W1.unaryExpr([&](double x) { return dist(gen); });
    MatrixXd b1(hidden_size, 1);
    b1 = b1.unaryExpr([&](double x) { return dist(gen); });
    MatrixXd W2(output_size, hidden_size);
    W2 = W2.unaryExpr([&](double x) { return dist(gen); });
    MatrixXd b2(output_size, 1);
    b2 = b2.unaryExpr([&](double x) { return dist(gen); });

    return {W1, b1, W2, b2};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> forward_propagation(const MatrixXd &X, const MatrixXd &W1, const MatrixXd &b1, const MatrixXd &W2, const MatrixXd &b2) {
    MatrixXd Z1 = W1 * X + b1.replicate(1, X.cols());
    MatrixXd A1 = sigmoid(Z1);
    MatrixXd Z2 = W2 * A1 + b2.replicate(1, A1.cols());
    MatrixXd A2 = sigmoid(Z2);

    return {Z1, A1, Z2, A2};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> backward_propagation(const MatrixXd &X, const MatrixXd &Y, const MatrixXd &Z1, const MatrixXd &A1, const MatrixXd &Z2, const MatrixXd &A2, const MatrixXd &W1, const MatrixXd &b1, const MatrixXd &W2, const MatrixXd &b2) {
    int m = X.cols();

    MatrixXd dZ2 = A2 - Y;
    MatrixXd dW2 = (1.0/m) * dZ2 * A1.transpose();
    MatrixXd db2 = (1.0/m) * dZ2.rowwise().sum();
    MatrixXd dZ1 = W2.transpose() * dZ2.array() * sigmoid_derivative(Z1).array();
    MatrixXd dW1 = (1.0/m) * dZ1 * X.transpose();
    MatrixXd db1 = (1.0/m) * dZ1.rowwise().sum();

    return {dW1, db1, dW2, db2};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> update_parameters(const MatrixXd &W1, const MatrixXd &b1, const MatrixXd &W2, const MatrixXd &b2, const MatrixXd &dW1, const MatrixXd &db1, const MatrixXd &dW2, const MatrixXd &db2, double learning_rate) {
    MatrixXd new_W1 = W1 - learning_rate * dW1;
    MatrixXd new_b1 = b1 - learning_rate * db1;
    MatrixXd new_W2 = W2 - learning_rate * dW2;
    MatrixXd new_b2 = b2 - learning_rate * db2;

    return {new_W1, new_b1, new_W2, new_b2};
}

std::tuple<MatrixXd, MatrixXd, MatrixXd, MatrixXd> train_nn(const MatrixXd &X, const MatrixXd &Y, int epochs, double learning_rate) {
    int input_size = X.rows();
    int hidden_size = 3;
    int output_size = 1;

    MatrixXd W1, b1, W2, b2;
    std::tie(W1, b1, W2, b2) = initialize_parameters(input_size, hidden_size, output_size);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        MatrixXd Z1, A1, Z2, A2;
        std::tie(Z1, A1, Z2, A2) = forward_propagation(X, W1, b1, W2, b2);
        MatrixXd dW1, db1, dW2, db2;
        std::tie(dW1, db1, dW2, db2) = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2);
        std::tie(W1, b1, W2, b2) = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate);
    }

    return {W1, b1, W2, b2};
}

int main() {
    MatrixXd X(2, 4);
    X << 0, 0, 1, 1,
         0, 1, 0, 1;
    MatrixXd Y(1, 4);
    Y << 0, 1, 1, 0;

    int epochs = 10000;
    double learning_rate = 0.1;

    MatrixXd W1, b1, W2, b2;
    std::tie(W1, b1, W2, b2) = train_nn(X, Y, epochs, learning_rate);

    MatrixXd Z1, A1, Z2, A2;
    std::tie(Z1, A1, Z2, A2) = forward_propagation(X, W1, b1, W2, b2);

    std::cout << "Predictions: " << std::endl;
    std::cout << A2 << std::endl;

    return 0;
}
