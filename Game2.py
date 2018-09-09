import pygame, sys
from pygame.locals import *
import numpy as np
import math

def drawBird(DISPLAYSURF, bird, pos):
    return DISPLAYSURF.blit(bird, pos)

def init_network():
    layout = [11, 3]
    # print("Initializing Network...")
    weights = list()
    count = 0
    for i in range(0, len(layout) - 1):
        # print("Weight%d: %d * %d" % (count, layout[i + 1], layout[i] + 1))
        temp_weight = (np.random.rand(layout[i + 1], layout[i] + 1)*2)-1
        # print(temp_weight)
        weights.append(temp_weight)
        count += 1
    return weights

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

def add_bias(x):
    x = np.append([1], x)
    return x

def getMove(x, weight):
    temp = add_bias(x)
    weight = np.array(weight)
    net = weight.dot(temp)
    out = sigmoid(net)
    # print(out)
    return np.argmax(out)

def drawCactus(DISPLAYSURF, cactus):
    a = DISPLAYSURF.blit(cactus, (WINDOWX, WINDOWY - 10 - cactus.get_rect().size[1]))
    a.width = cactus.get_rect().size[0]
    a.height = cactus.get_rect().size[1]
    return a

def drawPlayer(DISPLAYSURF, player, coords):
    a = DISPLAYSURF.blit(player, coords)
    # print(a)
    return a

def displayScore(DISPLAYSURF, score, iteration, individual, objects):
    myfont = pygame.font.SysFont("courier new ", 15)
    text = "Iteration: "+ str(iteration)+"   Individual: "+ str(individual)+ "  Score: "+str(score)+"   Objects:"+str(objects)
    label = myfont.render(text, 1, (0, 0, 0))
    DISPLAYSURF.blit(label, (0, 0))

def drawAllCacti(DISPLAYSURF, cactiObjects):
    # print(cactiObjects)
    for i in range(0, len(cactiObjects)):
        DISPLAYSURF.blit(cacti[cactiObjects[i][0]], (cactiObjects[i][1][0], cactiObjects[i][1][1]))

def init_game():
    global  bird1, bird2, trex1, trex2, trexOg, trexJump, trexDuck1, trexDuck2, WINDOWX, WINDOWY, cacti, cactiObjects, maxJumpHeight, WHITE, FPS, fpsClock
    pygame.init()

    FPS = 60
    WINDOWX = 600
    WINDOWY = 250
    fpsClock = pygame.time.Clock()

    DISPLAYSURF = pygame.display.set_mode((WINDOWX, WINDOWY), 0, 32)
    pygame.display.set_caption('Game 2')

    WHITE = (255, 255, 255)
    trex1 = pygame.image.load('./Game2Images/trex1.png')
    trex2 = pygame.image.load('./Game2Images/trex2.png')
    trexJump = pygame.image.load('./Game2Images/trex-jump.png')
    trexDuck1 = pygame.image.load('./Game2Images/trex-duck1.png')
    trexDuck2 = pygame.image.load('./Game2Images/trex-duck2.png')


    cacti = []
    cactiObjects = []

    cactus1 = pygame.image.load('./Game2Images/cactus1.png')
    cactus2 = pygame.image.load('./Game2Images/cactus2.png')
    cactus3 = pygame.image.load('./Game2Images/cactus3.png')
    cactus4 = pygame.image.load('./Game2Images/cactus4.png')
    cactus5 = pygame.image.load('./Game2Images/cactus5.png')

    bird1 = pygame.image.load('./Game2Images/bird1.png')
    bird2 = pygame.image.load('./Game2Images/bird2.png')

    # print(DISPLAYSURF.blit(cactus1, (WINDOWX, 10)).width)
    # print(DISPLAYSURF.blit(cactus2, (WINDOWX, 10)))
    # print(DISPLAYSURF.blit(cactus3, (WINDOWX, 10)))
    # print(cactus1.get_rect().size)

    cacti.append(cactus1)
    cacti.append(cactus2)
    cacti.append(cactus3)
    cacti.append(cactus4)
    # cacti.append(cactus5)

    imageSize = trex1.get_rect().size
    # print(imageSize)
    trexX = 100
    trexY = WINDOWY - 10 - imageSize[1]
    trexOg = WINDOWY - 10 - imageSize[1]

    maxJumpHeight = 85
    return DISPLAYSURF, cactiObjects, trexX, trexY, cacti

def startGame(weights, iteration, individual):
    DISPLAYSURF, cactiObjects, trexX, trexY, cacti= init_game()
    count = 0

    jumpHeight = 0
    isJumping = False
    jump = False
    duck = False
    isAscending = False
    isDescending = False
    gameSpeed = 5
    start_game = False

    numCactiOnscreen = 0
    maxCactiOnScreen = 5
    minDistanceBetweenCacti = 200
    jumpCount = 0
    objectsCrossed = 0
    score = 0
    bird = 0
    birdX = 0
    birdY = 0
    numBird = 0
    nearestObject = 0

    while True:
        DISPLAYSURF.fill(WHITE)
        pygame.draw.line(DISPLAYSURF, (128, 128, 128), (0, WINDOWY - 15), (WINDOWX, WINDOWY - 15), 1)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if not start_game:
            start_game = True

        for cactiObject in cactiObjects:
            if cactiObject[1][0] > trexX+player.size[0]:
                firstCactus = cactiObject
                break

        # print(trexY)
        if cactiObjects:
            if bird != 0:
                inputs = [firstCactus[1][0], firstCactus[1][1], firstCactus[1][2], firstCactus[1][3], bird[0], bird[1], bird[2], bird[3], trexY, trexY, gameSpeed]
            else:
                inputs = [firstCactus[1][0], firstCactus[1][1], firstCactus[1][2], firstCactus[1][3], 0, 0, 0, 0, trexY, trexY, gameSpeed]
            move = getMove(inputs, weights)

            if move == 0:
                # print("Do nothing")
                if not isJumping:
                    if duck:
                        duck = False
                        trexY-=15
                jump = False
            elif move == 1:
                # print("Jump")
                if duck:
                    duck = False
                    trexY -= 15
                jump = True
            elif move == 2:
                # print("Duck")
                jump = False
                if not isJumping:
                    duck = True
                    trexY = trexOg + 15

        if start_game:
            if (jump == True):
                isJumping = True

            if isJumping:
                if not isDescending:
                    isAscending = True
                    jumpHeight += gameSpeed
                    trexY -= gameSpeed

                    if jumpHeight >= maxJumpHeight:
                        isAscending = False
                        isDescending = True

                elif isDescending:
                    jumpHeight -= gameSpeed
                    trexY += gameSpeed

                    if jumpHeight <= 0:
                        isDescending = False
                        isJumping = False

            if duck:
                if int(count / 5) % 2 == 0:
                    player = drawPlayer(DISPLAYSURF, trexDuck1, (trexX, trexY))
                else:
                    player = drawPlayer(DISPLAYSURF, trexDuck2, (trexX, trexY))
            elif isJumping:
                player = drawPlayer(DISPLAYSURF, trexJump, (trexX, trexY))
            else:
                if int(count / 5) % 2 == 0:
                    player = drawPlayer(DISPLAYSURF, trex1, (trexX, trexY))
                else:
                    player = drawPlayer(DISPLAYSURF, trex2, (trexX, trexY))
            count = (count + 1) % 10


            if(count==0):
                score+=1
                if (score > 1 and score % 200 == 0 and gameSpeed < 10):
                    gameSpeed += 1
                if (score > 1 and score % 50 == 0 and minDistanceBetweenCacti > 100):
                    minDistanceBetweenCacti -= 5

            if numCactiOnscreen < maxCactiOnScreen:
                if (nearestObject != 0 and nearestObject[0]< WINDOWX - minDistanceBetweenCacti -nearestObject[2]) or (len(cactiObjects)==0 and numBird==0):
                    generate = np.random.randint(0, 1000)
                    if generate<30:
                        random = np.random.randint(0, len(cacti))
                        temp = [random, drawCactus(DISPLAYSURF, cacti[random])]
                        nearestObject = temp[1]
                        cactiObjects.append(temp)
                        numCactiOnscreen+=1
                    if generate>30 and generate<50 and score>100:
                        if numBird == 0:
                            numBird = 1
                            choices = [150, WINDOWY-50, 175]
                            random = np.random.randint(0, len(choices))
                            birdX = WINDOWX
                            birdY = choices[random]
                            bird = DISPLAYSURF.blit(bird1, (birdX, birdY))
                            nearestObject= bird

            cactiToRemove = []
            for cactiObject in cactiObjects:
                cactiObject[1].left -= gameSpeed
                if(cactiObject[1][0] < 0):
                    cactiToRemove.append(cactiObject)
                    numCactiOnscreen -= 1

            for i in range(0, len(cactiToRemove)):
                cactiObjects.remove(cactiToRemove[i])
                objectsCrossed+=1

            if numBird>0:
                if player.colliderect(bird):
                    return objectsCrossed, score
            for i in range(0, len(cactiObjects)):
                # print(cactiObjects[i][1], " , ", player)
                if player.colliderect(cactiObjects[i][1]):
                    # start_game =
                    # score  = startGame()
                    return objectsCrossed, score
            # if len(cactiObjects)>0:
            #     print(cactiObjects)
            #     # if player.collidelist(cactiObjects[1]):
            #     #  print("collision")

            # print(numBird)

            if numBird == 1:
                birdX -= gameSpeed
                if birdX <= 0:
                    numBird = 0
                    objectsCrossed+=1
                if int(count / 5) % 2 == 0:
                    bird = drawBird(DISPLAYSURF, bird1, (birdX, birdY-5))
                else:
                    bird = drawBird(DISPLAYSURF, bird2, (birdX, birdY))

        else:
            player = drawPlayer(DISPLAYSURF, trexJump, (trexX, trexY))
        drawAllCacti(DISPLAYSURF, cactiObjects)
        displayScore(DISPLAYSURF, score, iteration, individual, objectsCrossed)
        pygame.display.update()
        fpsClock.tick(FPS)

def printPopAndSelect(population, fitness, scores, timePlayed):
    print(population)
    eCount = []
    aCount = []
    prob = []
    total = sum(fitness)
    avg = total / POPULATION_SIZE
    # maximum = max(fitness)

    # print(population)
    fitness = np.array(fitness)
    population = np.array(population)
    scores = np.array(scores)
    timePlayed = np.array(timePlayed)

    inds = fitness.argsort()
    inds[:] = inds[::-1]
    fitness = fitness[inds]
    population = population[inds]
    scores = scores[inds]
    timePlayed = timePlayed[inds]

    # print("score \t time \t f(x) \t f(x)/sum \t f(x)/avg \t count")
    print("\n")
    for i in range(0, POPULATION_SIZE):
        prob.append(fitness[i] / total)
        eCount.append(fitness[i] / avg)
        aCount.append(int(round(eCount[i])))
        # print(population[i].tolist(), "\t", str(scores[i]).zfill(3), "\t", round(timePlayed[i]/300, 3), "\t", round(fitness[i], 3),"\t",    round(prob[i], 3), "\t\t", round(eCount[i], 3), "\t\t", aCount[i])
        print(str(scores[i]).zfill(3), "\t",str(timePlayed[i]).zfill(3), "\t", round(fitness[i], 3),"\t",    round(prob[i], 3), "\t\t", round(eCount[i], 3), "\t\t", aCount[i], "\t", population[i].tolist())

    print("Fitness is: ", sum(fitness), "\n")
    t = []

    for i in range(0, POPULATION_SIZE):
        for j in range(0, aCount[i]):
            t.append(population[i])
    population = t

    if (sum(aCount) > POPULATION_SIZE):
        population = population[:POPULATION_SIZE]
    if (sum(aCount) < POPULATION_SIZE):
        diff = POPULATION_SIZE - sum(aCount)
        for i in range(0, diff):
            population.append(population[i])
    # print(population)
    return population

def crossover(initial_population):
    population = []
    # print(initial_population)
    np.random.shuffle(initial_population)

    temp = []
    for i in range(0, POPULATION_SIZE):
        temp.append(initial_population[i].tolist()[0])

    initial_population = temp

    for i in range(0, int(POPULATION_SIZE/2)):
        if(np.random.randint(0, 100) < CROSSOVER_RATE*100):
            # print(len(initial_population[0]))
            crossover_point = np.random.randint(0, len(initial_population[0]))
            chr1 = []
            chr2 = []
            for j in range(0, crossover_point):
                chr1.append(initial_population[i*2][j])
                chr2.append(initial_population[(i*2)+1][j])
            for j in range(crossover_point, len(initial_population[0])):
                chr1.append(initial_population[(i*2)+1][j])
                chr2.append(initial_population[(i*2)][j])
            population.append(chr1)
            population.append(chr2)
        else:
            # print("Dont Crossover")
            population.append(initial_population[i*2])
            population.append(initial_population[(i*2)+1])
    # print(population)
    return population

def mutation(initial_population):
    population = []
    # print(initial_population)
    for i in range(0, POPULATION_SIZE):
        if(np.random.randint(0, 100) < MUTATION_RATE*100):
            # print("Mutate")
            ch = []
            mutation_point = []
            num_mutations = np.random.randint(1, MAX_MUTATION)
            for k in range(0, num_mutations):
                mutation_point.append(np.random.randint(1, len(initial_population[0])))
            # print(mutation_point)
            # print(len(initial_population[0]))
            for j in range(0, len(initial_population[0])):
                if(not (j in mutation_point)):

                    ch.append(initial_population[i][j])
                else:
                    # print((np.random.randn()*2)-1)
                    ch.append((np.random.randn()*2)-1)
            # print(ch)
            # print(ch)
            population.append(ch)
        else:
            # print("Dont Mutate")
            # print(initial_population[i])
            population.append(initial_population[i])
    # print(population)
    return population


MAX_MUTATION = 5
MUTATION_RATE = 0.4
POPULATION_SIZE = 10
CROSSOVER_RATE = 0.9
ITERATIONS = 100
Generation = 36

weights = []
scores = []
fitness = []
timePlayed = []
dim = (3, 12)
allweights = []
population = []

for i in range(Generation, ITERATIONS):
    print("\n\nGeneration ", i)
    for j in range(0, POPULATION_SIZE):


        if i==Generation:
            weights = init_network()[0]
        else:
            weights = allweights[j]
        population.append(weights)

        print("Using: ", weights.tolist())
        score, time = startGame(weights=weights, iteration=i, individual=j)
        scores.append(score)
        timePlayed.append(time)
        fitness.append(3*score + time/100)

    # print("Shape of population: ", population.shape)
    # pop1 = np.array(population)
    # print(pop1.tolist())
    allweights.clear()
    for weight in population:
        # print("Shape of weight: ", weight.shape)
        weight = np.reshape(weight, (1, weight.shape[0] * weight.shape[1]))
        # print("Shape of weight: ", weight.shape)
        weight = weight.tolist()
        allweights.append(weight)

    pop1 = np.array(population)
    print("Before Selection\n", pop1.tolist())
    population = printPopAndSelect(allweights, fitness, scores, timePlayed=timePlayed)

    pop1 = np.array(population)
    print("After Selection\n", pop1.tolist())

    pop1 = np.array(population)
    print("Before Crossover\n", pop1.tolist())

    population = crossover(population)

    print("After Crossover\n", population)

    population = mutation(population)

    pop1 = np.array(population)
    print("After Mutation\n", pop1.tolist())
    # print(population)
    allweights.clear()
    for weight in population:
        # print(weight)
        weight = np.array(weight).reshape(dim)
        # print("Shape of weight: ", weight.shape)
        allweights.append(weight)

    population.clear()
    fitness.clear()
    scores.clear()

# score = startGame()
# print(score)
