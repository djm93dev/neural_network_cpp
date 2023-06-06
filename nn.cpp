#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

class NeuralNetwork {
private:
    int num_inputs;
    int num_hidden;
    int num_outputs;
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> hidden_outputs;
    std::vector<double> output_outputs;

public:
    NeuralNetwork(int num_inputs, int num_hidden, int num_outputs) {
        this->num_inputs = num_inputs;
        this->num_hidden = num_hidden;
        this->num_outputs = num_outputs;

        weights.resize(num_hidden + num_outputs);
        biases.resize(num_hidden + num_outputs);

        for (int i = 0; i < num_hidden; i++) {
            weights[i].resize(num_inputs);
            for (int j = 0; j < num_inputs; j++) {
                weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }

        for (int i = num_hidden; i < num_hidden + num_outputs; i++) {
            weights[i].resize(num_hidden);
            for (int j = 0; j < num_hidden; j++) {
                weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }

        for (int i = 0; i < num_hidden; i++) {
            biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }

        for (int i = num_hidden; i < num_hidden + num_outputs; i++) {
            biases[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        return x * (1 - x);
    }

    std::vector<double> feed_forward(const std::vector<double>& inputs) {
        hidden_outputs.clear();
        output_outputs.clear();

        for (int i = 0; i < num_hidden; i++) {
            double sum = 0;
            for (int j = 0; j < num_inputs; j++) {
                sum += inputs[j] * weights[i][j];
            }
            sum += biases[i];
            double output = sigmoid(sum);
            hidden_outputs.push_back(output);
        }

        for (int i = 0; i < num_outputs; i++) {
            double sum = 0;
            for (int j = 0; j < num_hidden; j++) {
                sum += hidden_outputs[j] * weights[i + num_hidden][j];
            }
            sum += biases[i + num_hidden];
            double output = sigmoid(sum);
            output_outputs.push_back(output);
        }

        return output_outputs;
    }

    void backpropagation(const std::vector<double>& inputs, const std::vector<double>& targets) {
        std::vector<double> output_errors(num_outputs);
        std::vector<double> hidden_errors(num_hidden);

        for (int i = 0; i < num_outputs; i++) {
            double error = targets[i] - output_outputs[i];
            output_errors[i] = error;
        }

        for (int i = 0; i < num_hidden; i++) {
            double sum = 0;
            for (int j = 0; j < num_outputs; j++) {
                sum += output_errors[j] * weights[j + num_hidden][i];
            }
            hidden_errors[i] = sum;
        }

        // Update weights and biases
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_inputs; j++) {
                weights[i][j] += hidden_errors[i] * sigmoid_derivative(hidden_outputs[i]) * inputs[j];
            }
            biases[i] += hidden_errors[i] * sigmoid_derivative(hidden_outputs[i]);
        }

        for (int i = 0; i < num_outputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                weights[i + num_hidden][j] += output_errors[i] * sigmoid_derivative(output_outputs[i]) * hidden_outputs[j];
            }
            biases[i + num_hidden] += output_errors[i] * sigmoid_derivative(output_outputs[i]);
        }
    }

    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < training_inputs.size(); i++) {
                const auto& inputs = training_inputs[i];
                const auto& targets = training_targets[i];

                // Feed forward
                feed_forward(inputs);

                // Backpropagation
                backpropagation(inputs, targets);
            }
        }
    }
    
    void save_weights(const std::string& filename) {
        std::ofstream file(filename);
        if (file.is_open()) {
            // Save the dimensions of the network
            file << num_inputs << " " << num_hidden << " " << num_outputs << std::endl;

            // Save the weights
            for (int i = 0; i < num_hidden + num_outputs; i++) {
                for (int j = 0; j < weights[i].size(); j++) {
                    file << weights[i][j] << " ";
                }
                file << std::endl;
            }

            // Save the biases
            for (int i = 0; i < num_hidden + num_outputs; i++) {
                file << biases[i] << " ";
            }

            file.close();
            std::cout << "Weights saved to file: " << filename << std::endl;
        }
        else {
            std::cout << "Failed to save weights to file: " << filename << std::endl;
        }
    }

    void load_weights(const std::string& filename) {
        std::ifstream file(filename);
        if (file.is_open()) {
            // Load the dimensions of the network
            int loaded_num_inputs, loaded_num_hidden, loaded_num_outputs;
            file >> loaded_num_inputs >> loaded_num_hidden >> loaded_num_outputs;

            // Verify compatibility of loaded weights with the current network
            if (loaded_num_inputs != num_inputs || loaded_num_hidden != num_hidden || loaded_num_outputs != num_outputs) {
                std::cout << "Error: Incompatible network dimensions in loaded weights file." << std::endl;
                return;
            }

            // Load the weights
            for (int i = 0; i < num_hidden + num_outputs; i++) {
                for (int j = 0; j < weights[i].size(); j++) {
                    file >> weights[i][j];
                }
            }

            // Load the biases
            for (int i = 0; i < num_hidden + num_outputs; i++) {
                file >> biases[i];
            }

            file.close();
            std::cout << "Weights loaded from file: " << filename << std::endl;
        }
        else {
            std::cout << "Failed to load weights from file: " << filename << std::endl;
        }
    }
};

int main() {
    int num_inputs = 2;
    int num_hidden = 3;
    int num_outputs = 1;

    NeuralNetwork neural_network(num_inputs, num_hidden, num_outputs);

    std::ifstream weights_file("weights.txt");
    if (weights_file) {
        // File exists, load the weights
        neural_network.load_weights("weights.txt");
        std::cout << "Loaded weights from file." << std::endl;
    }
    else {
        // File doesn't exist, train the network
        std::cout << "Weights file not found. Training the network..." << std::endl;

        std::vector<std::vector<double>> training_inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        // OR
        // std::vector<std::vector<double>> training_targets = {
        //     {0},
        //     {1},
        //     {1},
        //     {1}
        // };

        // AND
        // std::vector<std::vector<double>> training_targets = {
        //     {0},
        //     {0},
        //     {0},
        //     {1}
        // };

        // XOR
        std::vector<std::vector<double>> training_targets = {
            {0},
            {1},
            {1},
            {0}
        };

        // NOR
        //std::vector<std::vector<double>> training_targets = {
        //    {1},
        //    {0},
        //    {0},
        //    {0}
        //};

        neural_network.train(training_inputs, training_targets, 1000);

        // Save the weights to a file
        neural_network.save_weights("weights.txt");
        std::cout << "Weights saved to file." << std::endl;
    }

    // Test the network
    std::vector<double> test_input = { 0, 0 };
    std::vector<double> test_output = neural_network.feed_forward(test_input);
    std::cout << "0, 0: " << test_output[0] << std::endl;

    test_input = { 0, 1 };
    test_output = neural_network.feed_forward(test_input);
    std::cout << "0, 1: " << test_output[0] << std::endl;

    test_input = { 1, 0 };
    test_output = neural_network.feed_forward(test_input);
    std::cout << "1, 0: " << test_output[0] << std::endl;

    test_input = { 1, 1 };
    test_output = neural_network.feed_forward(test_input);
    std::cout << "1, 1: " << test_output[0] << std::endl;

    return 0;
}

