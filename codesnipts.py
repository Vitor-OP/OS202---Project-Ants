import pygame
pygame.init()

screen = pygame.display.set_mode((320,200))

cases_img = pygame.image.load("cases.png").convert_alpha()

screen.fill((255,255,255))
screen.blit(cases_img, (0,0))

pygame.display.update()

cases_spr = []
for i in range(0, 128, 8):
    cases_spr.append(pygame.Surface.subsurface(cases_img, i, 0, 8, 8))
    
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit(0)