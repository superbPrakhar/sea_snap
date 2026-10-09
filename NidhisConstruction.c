#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Command {
    int oldCube;
    int newCube;
    char dir[10];
};

int compare(const void *a, const void *b) {
    struct Command *x = (struct Command *)a;
    struct Command *y = (struct Command *)b;

    if (x->oldCube != y->oldCube)
        return x->oldCube - y->oldCube;

    return x->newCube - y->newCube;
}

int main() {
    int n;
    scanf("%d", &n);

    struct Command cmd[n];

    for (int i = 0; i < n; i++) {
        scanf("%d %d %s",
              &cmd[i].oldCube,
              &cmd[i].newCube,
              cmd[i].dir);
    }

    int target;
    scanf("%d", &target);

    int x[51] = {0};
    int y[51] = {0};
    int present[51] = {0};

    x[1] = 0;
    y[1] = 0;
    present[1] = 1;

    qsort(cmd, n, sizeof(struct Command), compare);

    for (int i = 0; i < n; i++) {
        int old = cmd[i].oldCube;
        int newCube = cmd[i].newCube;

        int nx = x[old];
        int ny = y[old];

        if (strcmp(cmd[i].dir, "right") == 0)
            nx++;
        else if (strcmp(cmd[i].dir, "left") == 0)
            nx--;
        else if (strcmp(cmd[i].dir, "top") == 0)
            ny++;
        else if (strcmp(cmd[i].dir, "down") == 0)
            ny--;

        for (int j = 1; j <= 50; j++) {
            if (present[j] && x[j] == nx && y[j] == ny) {
                present[j] = 0;
                break;
            }
        }

        x[newCube] = nx;
        y[newCube] = ny;
        present[newCube] = 1;
    }

    int top = -1;
    int down = -1;
    int left = -1;
    int right = -1;

    for (int i = 1; i <= 50; i++) {
        if (!present[i])
            continue;

        if (x[i] == x[target] && y[i] == y[target] + 1)
            top = i;
        else if (x[i] == x[target] && y[i] == y[target] - 1)
            down = i;
        else if (x[i] == x[target] - 1 && y[i] == y[target])
            left = i;
        else if (x[i] == x[target] + 1 && y[i] == y[target])
            right = i;
    }

    printf("%d %d %d %d", top, down, left, right);

    return 0;
}