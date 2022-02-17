# graph size settings
TILE_SIZE = 50

WIDTH = 800
HEIGHT = 800
BORDER_THICKNESS = 2

ROWS = HEIGHT // TILE_SIZE
COLS = WIDTH // TILE_SIZE


# visual settings

# colours
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 127, 0)
PURPLE = (255, 0, 255)

# graph colours
OUTLINE_COLOUR = BLACK
EMPTY_COLOUR = WHITE
BARRIER_COLOUR = BLACK

START_COLOUR = BLUE
END_COLOUR = ORANGE

CLOSED_COLOUR = RED
OPEN_COLOUR = GREEN

PATH_COLOUR = PURPLE

ALGORITHM = "DJIKSTRA"

# ALGORITHM = "A STAR"
# H = "EUCLIDEAN"