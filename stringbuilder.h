#ifndef __STRING_BUILDER_H__
#define __STRING_BUILDER_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "dynamic_array.h"

typedef struct {
    char *items;
    size_t count;
    size_t capacity;
} String_Builder;

void sb_append(String_Builder *sb, const char *str);
char *sb_to_cstr(String_Builder *sb);
void sb_reset(String_Builder *sb);
void sb_free(String_Builder *sb);

void sb_append(String_Builder *sb, const char *str)
{
    da_append_many(sb, str, strlen(str));
}

void sb_reset(String_Builder *sb)
{
    da_resize(sb, 0);
}

void sb_free(String_Builder *sb)
{
    da_free(sb);
}

char *sb_to_cstr(String_Builder *sb)
{
    da_append(sb, '\0');
    return sb->items; 
}

#endif
