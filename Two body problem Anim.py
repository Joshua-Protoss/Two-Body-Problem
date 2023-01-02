import pygame
import sys
import numpy as np
from numpy import sin, cos, tan, pi
from numpy.linalg import inv
from pygame.locals import *
from spring import spring

# functions and class
def G(y,t):
    
    x_d, theta_d, = y[0], y[1]
    x, theta = y[2], y[3]
    x_dd = x*theta_d**2 - 4*k*(x-x0)/m
    theta_dd = -2*theta_d*x_d/x
    
    return np.array([x_dd, theta_dd, x_d, theta_d])

def RK4_step(y, t, dt):
    k1 = G(y, t)
    k2 = G(y+0.5*k1*dt, t+0.5*dt)
    k3 = G(y+0.5*k2*dt, t+0.5*dt)
    k4 = G(y+k3*dt, t+dt)
    
    return dt * (k1 + 2*k2 + 2*k3 + k4)/6

def update(x, theta):
    x_coord = scale * (x) * sin(theta) + offset[0]
    y_coord = scale * (x) * cos(theta) + offset[1]
 
    return (int(x_coord), int(y_coord))

def update2(x, theta):
    x_coord = scale * (-x) * sin(theta) + offset[0]
    y_coord = scale * (-x) * cos(theta) + offset[1]
 
    return (int(x_coord), int(y_coord))

def render2(point):
    x, y = point[0], point[1]
    pygame.draw.circle(screen, BLUE, (x,y), int(m*4))
    return (x, y)

def render(point):
    x, y = point[0], point[1]
    
    if prev_point:
        pygame.draw.line(trace, LT_BLUE, prev_point, (x,y), 5)
    
    screen.fill(WHITE)
    if is_tracing:
        screen.blit(trace, (0,0))
    
    s.update(point2, point)
    s.render()
    pygame.draw.circle(screen, BLACK, offset, 10)
    pygame.draw.circle(screen, RED, (x,y), int(m*4))
    
    return (x, y)

class Spring():
    def __init__(self, color, start, end, nodes, width, lead1, lead2):
        self.start = start
        self.end = end
        self.nodes = nodes
        self.width = width
        self.lead1 = lead1
        self.lead2 = lead2
        self.weight = 3
        self.color = color

    def update(self, start, end):
        self.start = start
        self.end = end
        self.x, self.y, self.p1, self.p2 = spring(self.start, self.end, self.nodes, self.width, self.lead1, self.lead2)
        self.p1 = (int(self.p1[0]), int(self.p1[1]))
        self.p2 = (int(self.p2[0]), int(self.p2[1]))
        
    def render(self):
        pygame.draw.line(screen, self.color, self.start, self.p1, self.weight)
        prev_point = self.p1
        for point in zip(self.x, self.y):
            pygame.draw.line(screen, self.color, prev_point, point, self.weight)
            prev_point = point
        pygame.draw.line(screen, self.color, self.p2, self.end, self.weight)

# Pygame setup
w, h = 1024, 768
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
BLUE = (0,0,255)
LT_BLUE = (230,230,255)
offset = (800,400)
scale = 100

is_tracing = True
prev_point = None

screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
screen.fill(WHITE)
trace = screen.copy()
pygame.display.update()
clock = pygame.time.Clock()

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 38)

s = Spring(BLACK, (0,0), (0,0), 25, 30, 65, 65)

# parameters
m = 8.0
x0 = 1.0
k = 10.5
t = 0.0
delta_t = 0.01
y = np.array([0.0, 0.65, 3.0, 0.2])

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_t:
                is_tracing = not(is_tracing)
            if event.key == K_c:
                trace.fill(WHITE)
            
    point = update(y[2],y[3])
    point2 = update2(y[2],y[3])
    prev_point = render(point)
    render2(point2)

    time_string = 'Time: {} seconds'.format(round(t,1))
    text = myfont.render(time_string, False, (0,0,0))
    screen.blit(text,(10,10))

    t += delta_t
    y = y + RK4_step(y, t, delta_t)
    
    clock.tick(60)
    pygame.display.update()
