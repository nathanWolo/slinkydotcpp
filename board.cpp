#include <iostream>
#include <vector>

const int BOARD_WIDTH = 12;
const int BOARD_HEIGHT = 12;

enum class CellType : u_int8_t {
    Empty = 0,
    Food = 0xF,
    Snake0 = 0b00000001 ,
    Snake1 = 0b01000001,
    Snake2 = 0b10000001,
    Snake3 = 0b11000001,
};

class Board {

    public:
        Board(int width, int height)
            : width(width), height(height),
            grid(height, std::vector<CellType>(width, CellType::Empty)) {}

        void setCellValue(int x, int y, CellType value) {
            grid[y][x] = value;
        }

        void render() const {
        // Print column numbers
        std::cout << "  ";
        for (int i = 0; i < width; ++i) {
            std::cout << i % width << "  ";
        }
        std::cout << std::endl;

        // Print the board along with row numbers
        for (int y = 0; y < height; ++y) {
            // Print row number
            std::cout << y % 10 << " ";

            // Print cells
            for (const auto& cell : grid[y]) {
                char symbol = ' ';
                switch(cell) {
                    case CellType::Empty:
                        symbol = ' ';
                        break;
                    // Handle other cases as needed
                    case CellType::Food:
                        symbol = 'F';
                        break;
                    case CellType::Snake0:
                        symbol = '0';
                        break;
                    case CellType::Snake1:
                        symbol = '1';
                        break;
                    case CellType::Snake2:
                        symbol = '2';
                        break;
                    case CellType::Snake3:
                        symbol = '3';
                        break;
                }
                std::cout << "[" << symbol << "]";
            }
            std::cout << std::endl;
        }

        // Print column numbers again for better readability
        std::cout << "  ";
        for (int i = 0; i < width; ++i) {
            std::cout << i % width << "  ";
        }
        std::cout << std::endl;
    }


    private:
        int width, height;
        std::vector<std::vector<CellType>> grid;
};

int main() {
    Board board(BOARD_WIDTH, BOARD_HEIGHT);
    board.setCellValue(0, 0, CellType::Snake0);
    board.setCellValue(1, 0, CellType::Snake1);
    board.setCellValue(2, 0, CellType::Snake2);
    board.setCellValue(3, 0, CellType::Snake3);
    board.setCellValue(4, 0, CellType::Food);
    board.render();
    return 0;
}
