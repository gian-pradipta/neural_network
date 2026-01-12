#define NOB_IMPLEMENTATION
#include "nob.h"

int main(int argc, char **argv)
{
    NOB_GO_REBUILD_URSELF(argc, argv);
    Nob_Cmd cmd = {0};
    nob_cmd_append(&cmd, "gcc", "-o", "./bin/main", "main.c");
    if (!nob_cmd_run(&cmd)) return 1;
    nob_cmd_append(&cmd, ".\\bin\\main.exe");
    if (!nob_cmd_run(&cmd)) return 1;
    return 0;
}
