import pygame, sys
from pygame.locals import *
import numpy as np
import math
import time as t


def init_network():
    layout = [7, 3]
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

def draw_player(DISPLAYSURF):
    player = pygame.draw.rect(DISPLAYSURF, (255, 255, 255), (playerX, playerY, playerWidth, playerHeight))
    return player

def draw_ball(DISPLAYSURF):
    ball = pygame.draw.rect(DISPLAYSURF, (255, 255, 255), (ballX, ballY, ballSize, ballSize))
    return ball

def displayScore(DISPLAYSURF, score, boardNum, iteration, individual):
    myfont = pygame.font.SysFont("courier new ", 15)
    text = "Iteration: " + str(iteration) +"   Individual: "+ str(individual) + "   Board: "+ str(boardNum) +"  Score: "+str(score)
    label = myfont.render(text, 1, (255, 255, 255))
    DISPLAYSURF.blit(label, (0, 0))

def drawBlocks(DISPLAYSURF, boxesToDraw=[]):
    if(len(boxesToDraw)!=0):
        for box in boxesToDraw:
            pygame.draw.rect(DISPLAYSURF,(128, 128, 128), box)
        return boxesToDraw

    startX = blockMargin
    startY = blockMargin + 20
    moreBlocks = True
    boxes = []
    blockWidth = np.random.randint(windowX/20, windowX/10)
    while(moreBlocks):
        temp_box = pygame.draw.rect(DISPLAYSURF, (128, 128, 128), (startX, startY,blockWidth , blockHeight))
        boxes.append(temp_box)
        if((windowX - startX)>2*(blockWidth + blockMargin)):
            startX+= blockWidth +blockMargin
        else:
            startX = blockMargin
            startY += blockMargin+blockHeight

        if(startY > 70):
            moreBlocks = False
    return boxes

def init_board(iteration, individual, board, score):

    FPS = 30

    pygame.init()
    FPSClock = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((windowX, windowY))
    pygame.display.set_caption('Game 1')
    player = draw_player(DISPLAYSURF)
    ball = draw_ball(DISPLAYSURF)
    boxes = drawBlocks(DISPLAYSURF)
    displayScore(DISPLAYSURF, score, board, iteration, individual)

    return DISPLAYSURF, FPS, FPSClock, player, boxes, ball

def startgame(weights, iteration, individual, board, score, timePlayed):
    global windowX, windowY, blockMargin, blockHeight, blockWidth, ballX, ballY, ballSize, playerX, playerY, playerWidth, playerHeight
    gameStarted = False

    totalscore = score
    gameScore = 0

    windowX = 500
    windowY = 400

    blockHeight = 20
    blockWidth = 20
    blockMargin = 2

    playerWidth = 60
    playerHeight = 10

    ballSize = 10

    playerX = (windowX-playerWidth)/2
    playerY = windowY-playerHeight

    ballX = (windowX-ballSize)/2
    ballY = (windowY-playerHeight-ballSize-blockMargin)

    directionX = -1
    directionY = -1

    moveRight = False
    moveLeft = False

    time = 0
    newGame = False
    gameComplete = False

    DISPLAYSURF, FPS, FPSClock, player, boxes, ball = init_board(iteration, individual, board, totalscore)
    totalBoxes = len(boxes)

    # print(weights)
    # weights = init_network()
    # print(weights)
    current_milli_time = lambda: int(round(t.time() * 1000))
    startTime = current_milli_time()
    # print(startTime)
    while True:
        # main game loop
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        if(not gameStarted):
            gameStarted = True

        # move = getMove([ballX, ballY, directionX, directionY, playerX, int(moveLeft), int(moveRight)], weights)
        nBallX = (windowX - ballX) / windowX
        nBallY = (windowY - ballY) / windowY
        nPlayerX = (windowX - playerX) / windowX
        move = getMove([nBallX, nBallY, directionX, directionY, nPlayerX, int(moveLeft), int(moveRight)], weights)

        # Do not move
        if move == 0:
            # print("Don not move")
            moveLeft = False
            moveRight = False

        # move left
        elif move == 1:
            # print("move left")
            moveLeft = True
            moveRight = False

        # move right
        elif move == 2:
            # print("move right")
            moveLeft = False
            moveRight = True

        if moveRight and playerX<(windowX-playerWidth):
            playerX+=1
        if moveLeft and playerX>0:
            playerX-=1



        if(gameStarted) :
            ballY+= directionY
            ballX+= directionX

        timePlayed += 1

        # print(count)

        if(ballX>=windowX-ballSize or ballX<=0):
            directionX *= -1
        elif(ballY<=0):
            directionY = 1
        elif ballY>=windowY:
            newGame = True
            gameComplete = False
        elif player.colliderect(ball) and (not moveLeft) and (not moveRight):
            directionY = -1
            ballY+= directionY
            ballX+= directionX
        elif player.colliderect(ball) and ((moveLeft) or (moveRight)):
            directionX *= -1
            directionY = -1
            ballY+= directionY
            ballX+= directionX
        else:
            for i in range(0, len(boxes)):
                if ball.colliderect(boxes[i]):
                    # print("collide with box")
                    totalscore += 1
                    gameScore += 1
                    time = 0
                    boxes.pop(i)
                    if(gameScore == totalBoxes):
                        newGame = True
                        gameComplete = True
                    directionX *= -1
                    directionY *= -1
                    break
        # print(boxes)
        if time > 30000:
            endTime = current_milli_time()
            return totalscore, timePlayed
        if newGame and gameComplete:
            totalscore, timePlayed = startgame(weights=weights, iteration=iteration, individual=individual, board=board + 1, score=totalscore, timePlayed=timePlayed)
        if newGame:
            # startgame(1, 0)
            endTime = current_milli_time()
            return totalscore, timePlayed

        DISPLAYSURF.fill((0, 0, 0))
        player = draw_player(DISPLAYSURF)
        ball = draw_ball(DISPLAYSURF)
        boxes = drawBlocks(DISPLAYSURF,boxesToDraw=boxes)
        displayScore(DISPLAYSURF, score=totalscore, boardNum=board, iteration=iteration, individual=individual)
        pygame.display.update()
        time += 1
        FPSClock.tick()

def printPopAndSelect(population, fitness, timePlayed, scores):
    eCount = []
    aCount = []
    prob = []
    total = sum(fitness)
    avg = total / POPULATION_SIZE
    # maximum = max(fitness)

    # print(population)
    fitness = np.array(fitness)
    timePlayed = np.array(timePlayed)
    population = np.array(population)
    scores = np.array(scores)

    print(population)
    inds = fitness.argsort()
    inds[:] = inds[::-1]
    fitness = fitness[inds]
    timePlayed = timePlayed[inds]
    population = population[inds]
    scores = scores[inds]

    # print("score \t time \t f(x) \t f(x)/sum \t f(x)/avg \t count")
    print("\n")
    for i in range(0, POPULATION_SIZE):
        prob.append(fitness[i] / total)
        eCount.append(fitness[i] / avg)
        aCount.append(int(round(eCount[i])))
        # print(population[i].tolist(), "\t", str(scores[i]).zfill(3), "\t", round(timePlayed[i]/300, 3), "\t", round(fitness[i], 3),"\t",    round(prob[i], 3), "\t\t", round(eCount[i], 3), "\t\t", aCount[i])
        print(str(scores[i]).zfill(3), "\t", round(timePlayed[i]/300, 3), "\t", round(fitness[i], 3),"\t",    round(prob[i], 3), "\t\t", round(eCount[i], 3), "\t\t", aCount[i], "\t", population[i].tolist())

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


MAX_MUTATION = 3
MUTATION_RATE = 0.1
POPULATION_SIZE = 10
CROSSOVER_RATE = 0.9
ITERATIONS = 1000

Generation = 1

weights = []
scores = []
timePlayed = []
fitness = []
dim = (3, 8)
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
        score, time = startgame(weights=weights, iteration=i, individual=j, board=1, score=0, timePlayed=0)
        scores.append(score)
        timePlayed.append(time)
        fitness.append(((time/30000) + (3*score))/2)

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
    population = printPopAndSelect(allweights, fitness, timePlayed, scores)

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
    timePlayed.clear()