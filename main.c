#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct Array{
    float** data;
    int rows;
    int cols;
};

struct NeuralNet{
    struct Array* weights;
    struct Array* deltas;
    struct Array* activations;
    struct Array* zs;
    struct Array* gradients;
    int num_layers;
    int* layer_sizes;
};

int multiply_matrices(struct Array* a, struct Array* b, struct Array* result);
int initXavier(struct Array* weights, int in_dim, int out_dim); 
int create_array(struct Array* arr, int rows, int cols);


int init_nn(struct NeuralNet* nn, int num_layers, int* layer_sizes){
    nn->num_layers = num_layers;
    nn->layer_sizes = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {
        nn->layer_sizes[i] = layer_sizes[i];
    }
    nn->weights = (struct Array*)malloc((num_layers - 1) * sizeof(struct Array));
    for (int i = 0; i < num_layers - 1; i++) {
        initXavier(&nn->weights[i], layer_sizes[i+1], layer_sizes[i]);
        create_array(&nn->deltas[i], layer_sizes[i+1], layer_sizes[i]);
        create_array(&nn->activations[i], layer_sizes[i + 1], 1);
        create_array(&nn->zs[i], layer_sizes[i + 1], 1);
        create_array(&nn->gradients[i], layer_sizes[i+1], layer_sizes[i]);
    }
    return 0;
}



int nn_feedforward(struct NeuralNet* nn, struct Array* input, struct Array* output) {
    struct Array current_input = *input;
    struct Array current_output;
    for (int i = 0; i < nn->num_layers - 1; i++) {
        multiply_matrices(&current_input, &nn->weights[i], &current_output);
        // Apply activation function (ReLU)
        for (int r = 0; r < current_output.rows; r++) {
            for (int c = 0; c < current_output.cols; c++) {
                if (current_output.data[r][c] < 0) {
                    current_output.data[r][c] = 0;
                }
            }
        }
        current_input = current_output;
    }
    // Apply softmax to final output layer
    float sum_exp = 0.0;
    for (int r = 0; r < current_output.rows; r++) {
        for (int c = 0; c < current_output.cols; c++) {
            current_output.data[r][c] = exp(current_output.data[r][c]);
            sum_exp += current_output.data[r][c];
        }
    }
    for (int r = 0; r < current_output.rows; r++) {
        for (int c = 0; c < current_output.cols; c++) {
            current_output.data[r][c] /= sum_exp;
        }
    }
    *output = current_output;

    return 0;
}

int nn_backpropagate(struct NeuralNet* nn, struct Array* input, int target){
    // Compute the errors
    struct Array output;
    nn_feedforward(nn, input, &output);
    
    // compute cross-log entropy for label vs output (vector)
    float loss = -log(output.data[target][0] + 1e-10);
    printf("Loss: %f\n", loss); 
    return 0;
}   




int create_array(struct Array* arr, int rows, int cols) {
    arr->rows = rows;
    arr->cols = cols;
    arr->data = (int**)malloc(rows * sizeof(int*));
    if (arr->data == NULL) return -1;
    for (int i = 0; i < rows; i++) {
        arr->data[i] = (int*)malloc(cols * sizeof(int));
        if (arr->data[i] == NULL) return -1;
    }
    return 0;
}




int load_data(const char* filename, struct  Array* images, struct Array* labels){
    FILE* file = fopen(filename, "rb");
    if(file == NULL) return -1;
    char buffer[8192];
    int row = 0;
    int col = 0;
if (fgets(buffer, sizeof(buffer), file) == NULL) {
        fclose(file);
        return -1;  // empty file
    }
    while(fgets(buffer, sizeof(buffer), file)){
        col = 0;
        char* token = strtok(buffer, ",");
        //first token is label
        if(labels != NULL){
            labels->data[row][0] = atof(token);
        }
        int label = atoi(token);
        //Next 784 tokens are pixel values
        while(token != NULL){
            token = strtok(NULL, ",");
            if(token != NULL){
                images->data[row][col] = atoi(token);
                col++;
            }
            if(col >= 784) break;
        }
        row++;
        if (row <= 5 || row % 5000 == 0) {
            printf("Loaded row %5d   label=%d\n", row, label);
        }
    }
    printf("After loading:\n");
    fclose(file);
    return 0;
}




int multiply_matrices(struct Array* a, struct Array* b, struct Array* result) {
    if (a->cols != b->rows) return -1;
    create_array(result, a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->data[i][j] = 0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    return 0;
}

int initXavier(struct Array* weights, int in_dim, int out_dim) {
    create_array(weights, in_dim, out_dim);
    float scale = sqrt(6.0 / (in_dim + out_dim));
    for (int i = 0; i < in_dim; i++) {
        for (int j = 0; j < out_dim; j++) {
            weights->data[i][j] = ((float)rand() / RAND_MAX) * 2 * scale - scale;
        }
    }
    return 0;
}



int main() {
    printf("Hello, World!\n");
    struct NeuralNet nn;
    int layer_sizes[] = {784, 128, 64, 10};
    init_nn(&nn, 4, layer_sizes);

    struct Array images;
    struct Array labels;
if (create_array(&images, 60000, 784) != 0) {
    fprintf(stderr, "Failed to allocate images array\n");
    return 1;
}
if (create_array(&labels, 60000, 1) != 0) {
    fprintf(stderr, "Failed to allocate labels array\n");
    return 1;
}
    /* relative to the project dir, so `make && ./main` works from anywhere */
    load_data("mnist_train.csv", &images, &labels);
    // Initialize input matrix with some values
    //
printf("After loading:\n");
printf("First label  : %f\n", labels.data[0][0]);   // should be 5
printf("Second label : %f\n", labels.data[1][0]);   // should be 0
printf("Third label  : %f\n", labels.data[2][0]);   // should be 4
    // show first 10 labelor(int j = 0; j < 28; j++){
    //
    //
    //
    //
    //
    //
const char* shades = " .:-=+*#%@";
int i = 0; // first image
for(int row = 0; row < 28; row++) {
    for(int col = 0; col < 28; col++) {
        int p = images.data[i][row * 28 + col];
        int idx = (p * 10) / 256;           // 0..9
        printf("%c", shades[idx]);
    }
    printf("\n");
}
    return 0;
}
