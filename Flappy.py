import pygame
import random
import numpy as np

WIDTH = 1280
HEIGHT = 720
GRAVITY = 1500
JUMP_STRENGTH = -500
WHITE = (255,255,255)



class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.size = (100, 100)
        self.big_image = pygame.image.load("Bird.png").convert()
        self.big_image.set_colorkey(WHITE)
        self.image = pygame.transform.scale(self.big_image, self.size)
        self.rect = pygame.Rect(0, 0, 70, 60)  # Initialize rect 
        self.velocity = 0
        self.reset()

    def draw(self):
        
        self.rect.topleft = (int(self.pos.x) + 13, int(self.pos.y) + 20)  # Update rect position

    def reset(self):
        self.pos = pygame.Vector2(200, HEIGHT / 2)
        self.velocity = 0 
        self.draw()  #####


    def flap(self):
        self.velocity = JUMP_STRENGTH
            

    def apply_gravity(self, dt):
        self.velocity += GRAVITY * dt
        self.pos.y += self.velocity * dt
        self.draw()




class Pipe(pygame.sprite.Sprite):
    def __init__(self,x,y,rotate,level):
        self.size = [100, 200]
        self.sizetuple = (100, 200)
        self.xposition = x 
        self.yposition = y #660
        self.image = pygame.image.load("Pipe.png")
        self.image = pygame.transform.scale(self.image, self.sizetuple)  # Scale the image
        self.upsidedown = rotate
        self.gap = 279 # gap size can change (hyperparamter)
        if self.upsidedown:
            self.rotate()
        
        self.imagecopy = self.image

        
        self.rect = pygame.Rect(0, 0, *self.sizetuple)
        self.speed = 5

        self.imagecopy = self.image
        if self.upsidedown == 0:
            self.size[1] = random.randint(0,380)   #change to make training harder. have to change gap size to account (ln55, ln69, ln106)
        else:
            self.size[1] = 660 - level - self.gap

        self.sizetuple = (self.size[0],self.size[1])
        self.stretch()


    def draw(self):
        if self.upsidedown:
            self.rect.topleft = (int(self.xposition), int(self.yposition))  # Update rect position
        else:
            self.rect.bottomleft = (int(self.xposition), int(self.yposition))  # Update rect position
        


    def update(self,level):
        self.xposition -= self.speed
        if self.xposition < -100:
            self.reset(level, 0)

        self.draw()
        

    def stretch(self):
        self.imagecopy = pygame.transform.scale(self.imagecopy, (self.sizetuple))
        self.rect = pygame.Rect(0, 0, *self.sizetuple)


    def rotate(self):     #
        self.image = pygame.transform.rotate(self.image, 180)

    def reset(self,level,x):
        if x == 0:
            self.xposition = 1500
        else:
            self.xposition = x 
        self.imagecopy = self.image
        if level == 0:
            self.size[1] = random.randint(0,380)    # if changing gap size copy size to this line also
        else:
            self.size[1] = 660 - level - self.gap

        self.sizetuple = (self.size[0],self.size[1])
        self.stretch()

    
class FlappyEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.background = pygame.transform.scale(pygame.image.load("Background.png"), (1280, 720))
        self.background_rect = pygame.Rect(0, 660, WIDTH, 1)
        self.ceiling_rect = pygame.Rect(0,0,WIDTH,1)
        self.font = pygame.font.Font('freesansbold.ttf',64)


        self.player = Player()
        #init pipes
        self.pipe1 = Pipe(1400, HEIGHT - 60, False, 0) 
        self.pipe2 = Pipe(1400, 0, True,self.pipe1.size[1])
        self.pipe3 = Pipe(2000, HEIGHT - 60, False, 0) 
        self.pipe4 = Pipe(2000, 0, True,self.pipe3.size[1])
        self.pipe5 = Pipe(2600, HEIGHT- 60, False, 0)
        self.pipe6 = Pipe(2600, 0, True,self.pipe5.size[1])

        self.pipes = [self.pipe1, self.pipe2, self.pipe3, self.pipe4, self.pipe5, self.pipe6]
        
        self.pipes_copy = self.pipes
        self.score = 0
        self.done = False
        self.dt = 1/60
        self.scored = False
        

    def reset(self):
        self.pipe_reset_pos = 1400
        self.score = 0 
        self.pipe_counter = 1

        self.done = False

        self.player.reset()
      
        self.pipe1.reset(0, 1400)
        self.pipe2.reset(self.pipe1.size[1], 1400)
        self.pipe3.reset(0, 2000)
        self.pipe4.reset(self.pipe3.size[1], 2000)
        self.pipe5.reset(0,2600)
        self.pipe6.reset(self.pipe5.size[1],2600)
        self.pipes = self.pipes_copy
        return self.get_state()      ##############


    def score_update(self):
        self.scored = False
        for pipe in self.pipes:
            if pipe in [self.pipe1, self.pipe3, self.pipe5]:  # Only count one pipe in the pair
                if pipe.xposition < 200 and pipe.xposition > 194 :  # Allow slight leeway because frame tick
                    self.score += 1
                    self.scored = True
        self.score_text = self.font.render(str(self.score), True, WHITE)
        self.screen.blit(self.score_text, ((WIDTH // 2) - 32 , 150)) #middle of screen 

        return self.score  

    


    def step(self, action,human):
        reward = 0.1
        self.scored = False
        # action 0 = do nothing 1 = flap
        if not human:
            if action == 1:
                self.player.flap()
                reward = -0.1    #small negative reward for flapping to discourage flapping all the time

                
            



        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if human:
                if event.type == pygame.KEYDOWN:  
                    if event.key == pygame.K_SPACE:  # space bar to jump if in human mode
                        self.player.flap()
        
        #physics update 
        self.player.apply_gravity(self.dt)
        self.pipe1.update(0)
        self.pipe2.update(self.pipe1.size[1])
        self.pipe3.update(0)
        self.pipe4.update(self.pipe3.size[1])
        self.pipe5.update(0)
        self.pipe6.update(self.pipe5.size[1])


        for pipe in self.pipes:
            if pipe in [self.pipe1, self.pipe3, self.pipe5]:  # Only count one pipe in the pair
                if pipe.xposition < 200 and pipe.xposition > 194 :  # Allow slight leeway
                    self.scored = True

        #rewards and termination 
        self.collide()
        if self.done:
           reward = -10.0  #big negative for dying
            
        elif self.scored:
            reward = 10.0 # big positive for going for pipe

        return self.get_state(), reward, self.done    ########################
        
    def render(self):

        self.screen.fill("white")
        self.screen.blit(self.background, (0, 0))

        self.score = self.score_update()
        self.screen.blit(self.player.image, (int(self.player.pos.x), int(self.player.pos.y)))

        for pipe in self.pipes:
            if pipe.upsidedown:
                self.screen.blit(pipe.imagecopy, (int(pipe.xposition), int(pipe.yposition)))
            else:
                self.screen.blit(pipe.imagecopy, (int(pipe.xposition), int(pipe.yposition)- pipe.size[1]))


       
        pygame.display.flip()
        self.clock.tick(60)        

    def collide(self):
        self.done = False #tracks epoch end
        if self.player.rect.colliderect(self.background_rect):
            self.done = True
        if self.player.rect.colliderect(self.ceiling_rect):
            self.done = True
        for pipe in self.pipes:
            if self.player.rect.colliderect(pipe.rect):
                self.done = True
   

    def get_state(self):
        # Find the next pipe pair in front of the player
        next_pipe = None
        for p in [self.pipe1, self.pipe3, self.pipe5]:
            if p.xposition + 100 > self.player.pos.x: # If the pipe's right edge is in front of the bird
                next_pipe = p
                break
        if next_pipe is None:
            next_pipe = self.pipe1

        #calculate gap
        gap_top = next_pipe.yposition - next_pipe.size[1] - next_pipe.gap
        gap_bottom = next_pipe.yposition - next_pipe.size[1]

        state = np.array([
            self.player.pos.y,              # Bird's Y-coordinate
            self.player.velocity,           # Bird's velocity
            next_pipe.xposition - self.player.pos.x,  # Horizontal distance to the pipe
            gap_top,                        # Y-coordinate of the top of the gap
            gap_bottom                      # Y-coordinate of the bottom of the gap
        ], dtype=np.float32)
        return state




#can play as yourself if you want to space to jump

"""
test = FlappyEnv()
while True:
    test.step(action=0, human=True)
    test.render()
"""