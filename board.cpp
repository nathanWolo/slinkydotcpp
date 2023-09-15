#include <iostream>

// Constants to define board dimensions
const int BOARD_WIDTH = 11;
const int BOARD_HEIGHT = 11;

void render_board() {
    for (int y = 0; y < BOARD_HEIGHT; ++y) {
        for (int x = 0; x < BOARD_WIDTH; ++x) {
            std::cout << "[]";
        }
        std::cout << std::endl;
    }
}

int main() {
    render_board();
    return 0;
}
