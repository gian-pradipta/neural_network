#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define NN_ARENA_IMPLEMENTATION 
#include "arena.h"
#undef NN_ARENA_IMPLEMENTATION 

#define NN_MAT_IMPLEMENTATION
#include "matrix.h"
#undef NN_MAT_IMPLEMENTATION

#include "nn_assert.h"
#include "dynamic_array.h"

#define NN_NAME nn 
#define NN_VERSION_MAJOR 0
#define NN_VERSION_MINOR 2 
#define NN_VERSION_PATCH 0

nn_arena arena = {0};

static nn_mat make_mat(size_t row, size_t col, float *es)
{
    nn_mat mat = {0};
    float *tbl = (float*) nn_arena_alloc(&arena, sizeof(float) * row * col);
    NN_ASSERT(tbl != NULL, "Buy new computer lmao");
    memcpy(tbl, es, sizeof(float) * row * col);
    nn_mat_init(&mat, row, col, tbl);
    return mat;
}

static nn_mat make_out(nn_mat *m1, nn_mat *m2)
{
    NN_ASSERT(m1 != NULL, "m1 is NULL");
    NN_ASSERT(m2 != NULL, "m2 is NULL");
    size_t r = m1->rows;
    size_t c = m2->cols;
    return make_mat(r, c, (float[]){0});
}

static inline float get_randf(float min, float max)
{

    return (float) rand() / (float) RAND_MAX;
    //return min + (float)rand() / (float)RAND_MAX * (max - min);
}

float sigmoidf(float x)
{
    return 1.0 / (1.0 + exp(-x));
}

float loss_mse(nn_mat *dataset)
{
    float sum = 0.0;
    for (size_t i = 0; i < dataset->rows; ++i) {
        float target = NN_MAT_AT(dataset, i, 2);
        float result = NN_MAT_AT(dataset, i, 3);
        sum += (target - result) * (target - result);
    }
    return sum / (float) dataset->rows;
}

nn_mat a;
nn_mat w;
nn_mat b;
nn_mat w2;
nn_mat b2;

float forward_pass(float x, float y)
{
    size_t checkpoint = arena.count;
    a = make_mat(1, 2,(float[]) {
            x, y
        });
    nn_mat o = make_out(&a, &w);
    nn_mat o2;

    nn_mat_mul(&a, &w, &o);
    o2 = make_out(&o, &b);
    nn_mat_add(&o, &b, &o2);
    nn_mat_map(&o2, sigmoidf, &o2);

    o = make_out(&o2, &w2);
    nn_mat_mul(&o2, &w2, &o);
    o2 = make_out(&o, &b2);
    nn_mat_add(&o, &b2, &o2);
    nn_mat_map(&o2, sigmoidf, &o2);
    float res = NN_MAT_AT(&o2, 0, 0);
    nn_arena_reset_to(&arena, checkpoint);
    return res;
}

void init_params(void)
{
    float min = -2.0;
    float max = 2.0;
    w = make_mat(2, 2,(float[]) {
            get_randf(min, max), get_randf(min, max),
            get_randf(min, max), get_randf(min, max),
            });
    b = make_mat(1, 2,(float[]) {
            get_randf(min, max), get_randf(min, max)
            });

    w2 = make_mat(2, 1,(float[]) {
            get_randf(min, max),
            get_randf(min, max),
            });

    b2 = make_mat(1, 1,(float[]) {
            get_randf(min, max)
            });
}

int main()
{
    srand(time(NULL));
    float mse; // f(x)
    float mse_2; // f(x + h)
    float gradient; // (f(x + h) - f(x)) / h
    // 256 MBs allocation.
    // Because memory is cheap and not even
    // Windows is stupid enough to give the 
    // entire 256 MBs blocks of memory to me lol.
    // The entire matrix allocation should go through
    // this point.
    nn_arena_init(&arena, 256 * 1000 * 1000);

    // Dataset for AND Truth Table. The 3rd
    // index is used to store the result values
    // of the Neural Network Model. The purpose
    // of this is to make it easier to calculate
    // the loss function.
    nn_mat dataset = make_mat(4, 4, (float[]){
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            1, 1, 1, 0
            });

    // For ultra-primitive gradient lookup of the 
    // Loss function to do gradient descent 
    float epsilon = 0.001;
    float learning_rate = 0.05;
    nn_mat dataset_2 = make_mat(4, 4, (float[]){
            0, 0, 0, 0,
            1, 0, 0, 0,
            0, 1, 0, 0,
            1, 1, 1, 0
            });

    init_params();
    size_t i = 0;
    size_t epoch = 10 * 1000;
    for (; i < epoch; ++i) {
        size_t memstate = arena.count;
        for (size_t j = 0; j < dataset.rows; ++j) {
            // INITIAL LOSS
            float res = forward_pass(NN_MAT_AT(&dataset, j, 0), NN_MAT_AT(&dataset, j, 1));
            NN_MAT_AT(&dataset, j, 3) = res;
        }

        for (size_t j = 0; j < dataset_2.rows; ++j) {
            size_t state = arena.count;
            nn_mat init_b = make_mat(b.rows, b.cols, b.es);
            nn_mat tmp_b = make_mat(b.rows, b.cols, b.es);
            nn_mat_fill(&tmp_b, epsilon);
            nn_mat_add(&b, &tmp_b, &b);

            nn_mat init_b2 = make_mat(b2.rows, b2.cols, b2.es);
            nn_mat tmp_b2 = make_mat(b2.rows, b2.cols, b2.es);
            nn_mat_fill(&tmp_b2, epsilon);
            nn_mat_add(&b2, &tmp_b2, &b2);
//
            nn_mat tmp_w = make_mat(w.rows, w.cols, w.es);
            nn_mat_fill(&tmp_w, epsilon);
            nn_mat_add(&w, &tmp_w, &w);

            nn_mat tmp_w2 = make_mat(w2.rows, w2.cols, w2.es);
            nn_mat_fill(&tmp_w2, epsilon);
            nn_mat_add(&w2, &tmp_w2, &w2);

            float res = forward_pass(NN_MAT_AT(&dataset_2, j, 0), NN_MAT_AT(&dataset_2, j, 1));
            NN_MAT_AT(&dataset_2, j, 3) = res;

            nn_mat_sub(&b, &tmp_b, &b);
            nn_mat_sub(&b2, &tmp_b2, &b2);
            nn_mat_sub(&w, &tmp_w, &w);
            nn_mat_sub(&w2, &tmp_w2, &w2);
            nn_arena_reset_to(&arena, state);
        }

        mse = loss_mse(&dataset); // f(x)
        mse_2 = loss_mse(&dataset_2); // f(x + h)
        gradient = (mse_2 - mse) / epsilon; // (f(x + h) - f(x)) / h

        nn_mat tmp = make_mat(w.rows, w.cols, w.es);
        nn_mat_fill(&tmp, (gradient * learning_rate));
        nn_mat_sub(&w, &tmp, &w);

        tmp = make_mat(w2.rows, w2.cols, w2.es);
        nn_mat_fill(&tmp, (gradient * learning_rate));
        nn_mat_sub(&w2, &tmp, &w2);

        if (i == 0) {
            printf("- ITERASI - 1 -\n");
            printf("finished with MSE = %f\n", mse);
            printf("finished with MSE2 = %f\n", mse_2);
            printf("finished with gradient = %f\n", gradient);
            printf("finished with learning = %f\n", gradient * learning_rate);
            printf("------------------------------\n\n");
        }
        tmp = make_mat(b.rows, b.cols, b.es);
        nn_mat_fill(&tmp, (gradient * learning_rate));
        nn_mat_sub(&b, &tmp, &b);

        tmp = make_mat(b2.rows, b2.cols, b2.es);
        nn_mat_fill(&tmp, (gradient * learning_rate));
        nn_mat_sub(&b2, &tmp, &b2);
        nn_arena_reset_to(&arena, memstate);
    }

    printf("- ITERASI - %zu -\n", i);
    printf("finished with MSE = %f\n", mse);
    printf("finished with MSE2 = %f\n", mse_2);
    printf("finished with gradient = %f\n", gradient);
    printf("finished with learning = %f\n", gradient * learning_rate);
    printf("------------------------------\n");
    nn_mat_print(&dataset_2);
    nn_mat_print(&w);
    return 0;
}

