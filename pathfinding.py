import pygame
import math
import settings
from queue import PriorityQueue

class Node:
    def __init__(self, x, y, row, col):
        self.x = x
        self.y = y
        self.row = row
        self.col = col
        self.colour = settings.EMPTY_COLOUR

        self.rect = pygame.Rect(x, y, settings.TILE_SIZE, settings.TILE_SIZE)
    
    def drawNode(self, screen):
        pygame.draw.rect(screen, self.colour, self.rect)
        pygame.draw.rect(screen, settings.OUTLINE_COLOUR, self.rect, settings.BORDER_THICKNESS)

    def makeStart(self):
        self.colour = settings.START_COLOUR
    
    def makeEnd(self):
        self.colour = settings.END_COLOUR

    def makeEmpty(self):
        self.colour = settings.EMPTY_COLOUR
    
    def makeBarrier(self):
        self.colour = settings.BARRIER_COLOUR

    def invertColour(self):
        if self.colour == settings.EMPTY_COLOUR:
            self.makeBarrier()
        
        else:
            self.makeEmpty()

    def makeClosed(self):
        self.colour = settings.CLOSED_COLOUR

    def makeOpen(self):
        self.colour = settings.OPEN_COLOUR
    
    def makePath(self):
        self.colour = settings.PATH_COLOUR

    def updateNeighbours(self, graph):
        self.neighbours = []
        
        if self.row > 0 and graph[self.row - 1][self.col].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row - 1][self.col]) # left
        
        if self.row < settings.ROWS - 1 and graph[self.row + 1][self.col].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row + 1][self.col]) # right
        
        if self.col > 0 and graph[self.row][self.col - 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row][self.col - 1]) # up
        
        if self.col < settings.COLS - 1 and graph[self.row][self.col + 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row][self.col + 1]) # down
        
        if self.row > 0 and self.col > 0 and graph[self.row - 1][self.col - 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row - 1][self.col - 1]) # up left
        
        if self.row < settings.ROWS - 1 and self.col > 0 and graph[self.row + 1][self.col - 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row + 1][self.col - 1]) # up right
        
        if self.row > 0 and self.col < settings.COLS - 1 and graph[self.row - 1][self.col + 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row - 1][self.col + 1]) # down left
        
        if self.row < settings.ROWS - 1 and self.col < settings.COLS - 1 and graph[self.row + 1][self.col + 1].colour != settings.BARRIER_COLOUR:
            self.neighbours.append(graph[self.row + 1][self.col + 1]) # down right
    
    def orthogonallyAdjacent(self, other):
        rows_dist = other.row - self.row
        cols_dist = other.col - self.col

        check_list = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        if (rows_dist, cols_dist) in check_list:
            return True
            
        return False
    
def makeGraph():
    graph = []
    for row in range(settings.ROWS):
        row_l = []

        for col in range(settings.COLS):
            row_l.append(Node(col * settings.TILE_SIZE, row * settings.TILE_SIZE, row, col))
        
        graph.append(row_l)

    return graph

def makePath(previous_nodes, current_node):
    while current_node in previous_nodes:
        current_node = previous_nodes[current_node]
        current_node.makePath()

        pygame.display.flip()
        pygame.time.delay(150)

def drawGraph(screen, graph):
    screen.fill(settings.WHITE)

    for row in graph:
        for node in row:
            node.drawNode(screen)
    
    pygame.display.flip()

def manhattanH(row_1, col_1, row_2, col_2):
    x_dist = abs(col_2 - col_1)
    y_dist = abs(row_2 - row_1)

    return x_dist + y_dist

def euclideanH(row_1, col_1, row_2, col_2):
    x_dist = col_2 - col_1
    y_dist = row_2 - row_1

    return math.sqrt(x_dist ** 2 + y_dist ** 2)

def djikstra(screen, graph, start_node, end_node):
    n = 0
    open_set = PriorityQueue()
    open_set.put((0, n, start_node)) # f score, node_n, node

    previous_nodes = {}

    g_scores = {node: float("inf") for row in graph for node in row}
    g_scores[start_node] = 0

    open_set_map = {start_node}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current_node = open_set.get()[2]
        open_set_map.remove(current_node)

        if current_node == end_node:
            drawGraph(screen, graph)
            pygame.time.delay(100)
            
            makePath(previous_nodes, end_node)

            start_node.makeStart()
            end_node.makeEnd()

            drawGraph(screen, graph)

            pygame.display.flip()
            pygame.time.delay(1000)

            return True
        
        for neighbour in current_node.neighbours:
            if current_node.orthogonallyAdjacent(neighbour):
                tentative_g_score = g_scores[current_node] + 1

            else:
                tentative_g_score = g_scores[current_node] + math.sqrt(2)
            
            if tentative_g_score < g_scores[neighbour]:
                previous_nodes[neighbour] = current_node

                g_scores[neighbour] = tentative_g_score

                if neighbour not in open_set_map:
                    n += 1
                    open_set.put((g_scores[neighbour], n, neighbour))
                    open_set_map.add(neighbour)

                    if neighbour != start_node and neighbour != end_node:
                        neighbour.makeOpen()
        
        drawGraph(screen, graph)
        pygame.time.delay(100)

        if current_node != start_node and current_node != end_node:
            current_node.makeClosed()

    return False

def aStar(screen, graph, start_node, end_node, h):
    n = 0
    open_set = PriorityQueue()
    open_set.put((0, n, start_node)) # f score, node_n, node

    previous_nodes = {}

    g_scores = {node: float("inf") for row in graph for node in row}
    g_scores[start_node] = 0

    f_scores = {node: float("inf") for row in graph for node in row}
    f_scores[start_node] = h(start_node.row, start_node.col, end_node.row, end_node.col)

    open_set_map = {start_node}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        
        current_node = open_set.get()[2]
        open_set_map.remove(current_node)

        if current_node == end_node:
            drawGraph(screen, graph)
            pygame.time.delay(100)
            
            makePath(previous_nodes, end_node)

            start_node.makeStart()
            end_node.makeEnd()
            drawGraph(screen, graph)

            pygame.display.flip()
            pygame.time.delay(1000)

            return True
        
        for neighbour in current_node.neighbours:
            if current_node.orthogonallyAdjacent(neighbour):
                tentative_g_score = g_scores[current_node] + 1

            else:
                tentative_g_score = g_scores[current_node] + math.sqrt(2)
            
            if tentative_g_score < g_scores[neighbour]:
                previous_nodes[neighbour] = current_node

                g_scores[neighbour] = tentative_g_score
                f_scores[neighbour] = tentative_g_score + h(neighbour.row, neighbour.col, end_node.row, end_node.col)

                if neighbour not in open_set_map:
                    n += 1
                    open_set.put((f_scores[neighbour], n, neighbour))
                    open_set_map.add(neighbour)

                    if neighbour != start_node and neighbour != end_node:
                        neighbour.makeOpen()
        
        drawGraph(screen, graph)
        pygame.time.delay(100)

        if current_node != start_node and current_node != end_node:
            current_node.makeClosed()

    return False

def clearGraph(graph):
    for row in graph:
        for node in row:
            node.makeEmpty()

def main():
    screen = pygame.display.set_mode((settings.WIDTH, settings.HEIGHT))
    
    graph = makeGraph()
    done = False
    
    start_node = None
    end_node = None
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()

                for row in graph:
                    for node in row:
                        if node.rect.collidepoint(pos):
                            if not start_node:
                                start_node = node
                                node.makeStart()

                            elif not end_node:
                                end_node = node
                                node.makeEnd()
                            
                            elif node != start_node and node != end_node:
                                node.invertColour()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start_node and end_node:
                    for row in graph:
                        for node in row:
                            node.updateNeighbours(graph)

                    if settings.ALGORITHM == "A STAR":
                        if settings.H == "EUCLIDEAN":
                            h = euclideanH
                            
                        elif settings.H == "MANHATTAN":
                            h = manhattanH

                        aStar(screen, graph, start_node, end_node, h)

                    elif settings.ALGORITHM == "DJIKSTRA":
                        djikstra(screen, graph, start_node, end_node)

                elif event.key == pygame.K_c:
                    clearGraph(graph)
                    start_node = None
                    end_node = None

        drawGraph(screen, graph)

if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()