class Cell():
    # A cell object knows about its location in the grid 
    # it also knows of its size with the variables x,y,w,h.
    def __init__(self, tempX, tempY, tempW, tempH, value):
        self.x = tempX
        self.y = tempY
        self.w = tempW
        self.h = tempH
        self.value = value
        self.visited = False
        
    def display(self):
        stroke(0)
        if self.visited:
            fill(136,231,136)
        else:
            fill(211,211,211)
        rect(self.x,self.y,self.w,self.h)
        if self.value:
            stroke(255, 165, 0)
            fill(255, 165, 0)
            ellipse(self.x+self.w/2,self.y+self.h/2,self.w/4,self.h/4)

            
class Ant():
    def __init__(self, x, y, path):
        self.x = x
        self.y = y
        self.path = path


GRID = [
    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    ]

ANT_PATH = [1, -1, 0, 0, 0, 1, -1, 0, -1, 1, 1, 0, 1, -1, 1, 0, -1, 0, -1, 1, 1, 1, -1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 1, 1, 1, -1, 0, 1, -1, 1, 1, 0, -1, 0, 0, -1, 1, 1, 0, -1]
ant = Ant(0, 0, ANT_PATH)
grid_length = 10

def setup():
    size(800, 800)
    draw_list(GRID)


def draw_list(grid):
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            grid[row][col]= Cell(col*80, row*80, 80, 80, grid[row][col]) 

def update_position(individual):
    global GRID, grid_length
    direction = 1  # 0=up, 1=right, 2=down, 3=left
    food_collected = 0
    x_pos, y_pos = 0, 0
    counted = []
    for move in individual:
        direction = (move + direction) % 4
        if direction == 0 and y_pos > 0:
            y_pos -= 1
        elif direction == 1 and x_pos < grid_length - 1:
            x_pos += 1
        elif direction == 2 and y_pos < grid_length - 1:
            y_pos += 1
        elif direction == 3 and x_pos > 0:
            x_pos -= 1
        if [y_pos, x_pos] not in counted:
            food_collected += 1
            counted.append([y_pos, x_pos])
    return counted
    

def draw():
    global GRID, ant
    background(211,211,211)
    ant_path = update_position(ant.path)
    for coordinate in ant_path:
        GRID[coordinate[0]][coordinate[1]].visited = True
    for i in range(len(GRID)):
        for j in range(len(GRID[0])):
            GRID[i][j].display()
