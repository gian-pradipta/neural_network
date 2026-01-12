#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NN_ARENA_IMPLEMENTATION 
#include "arena.h"
#undef NN_ARENA_IMPLEMENTATION 

#define NN_MAT_IMPLEMENTATION
#include "matrix.h"
#undef NN_MAT_IMPLEMENTATION

#include "nn_assert.h"

#define NAME nn 
#define VERSION_MAJOR 0
#define VERSION_MINOR 1 
#define VERSION_PATCH 0

nn_arena arena = {0};

static nn_mat make_mat(size_t row, size_t col, float *es)
{
    nn_mat mat = {0};
    float *tbl = (float*) nn_arena_alloc(&arena, sizeof(float) * row * col);
    if (tbl == NULL) {
        printf("Buy new computers lol");
        exit(1);
    }
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

int main()
{
    nn_arena_init(&arena, 256 * 1000 * 1000);
    nn_mat b = make_mat(1, 2,(float[]) {
                1,1
            });
    nn_mat m1 = make_mat(2, 3,(float[]) {
                1,0,0,
                0,1,0,
            });
    nn_mat m2 = make_mat(3, 3,(float[]) {
                1,1,1,
                1,-8,1,
                1,1,1
            });
    
    {
        nn_mat o = make_out(&m1, &m2);
        nn_mat_mul(&m1, &m2, &o);
        nn_mat_print(&o);
        o = make_out(&b, &o);
        nn_mat_mul(&b, &o, &o);
    }

    nn_arena_reset(&arena);
    printf("finished\n");
    return 0;
}
