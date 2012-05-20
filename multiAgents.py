# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = [action for action in gameState.getLegalActions() if action!='Stop']
    #print legalMoves
    
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where **higher numbers are better**.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    if successorGameState.isWin() :   return sys.maxint 
    if successorGameState.isLose() :  return -sys.maxint
    
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    #print '\nnewPos '+str(newPos)
    #print 'oldFood '+str(oldFood.asList())
    #print "score %d" % successorGameState.getScore()
    
    "obtain food score"
    newFood = successorGameState.getFood()
    newfoodList = newFood.asList()
    closestFood = min([util.manhattanDistance(newPos, foodPos) for foodPos in newfoodList])
    foodScore = 1.0 / closestFood
    
    "obtain ghost score"
    ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates if ghostState.scaredTimer == 0]
    if ghostPositions:
        closestGhost = min([util.manhattanDistance(newPos, ghostPos) for ghostPos in ghostPositions])
        if closestGhost == 0:
            return -sys.maxint
    else:
        return sys.maxint
    totalScaredTime = sum(newScaredTimes)    
    
    "a new evaluation function."
    heuristic = successorGameState.getScore() + foodScore/closestGhost + totalScaredTime
    #print heuristic
    return heuristic

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return self.MinimaxSearch(gameState, 1, 0 )

  def MinimaxSearch(self, gameState, currentDepth, agentIndex):
    "terminal check"
    if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    
    "minimax algorithm"
    legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
    
    # update next depth
    nextIndex = agentIndex + 1
    nextDepth = currentDepth
    if nextIndex >= gameState.getNumAgents():
        nextIndex = 0
        nextDepth += 1
    
    # Choose one of the best actions or keep query the minimax result
    results = [self.MinimaxSearch( gameState.generateSuccessor(agentIndex, action) ,\
                                  nextDepth, nextIndex) for action in legalMoves]
    if agentIndex == 0 and currentDepth == 1: # pacman first move
        bestMove = max(results)
        bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print 'pacman %d' % bestMove
        return legalMoves[chosenIndex]
    
    if agentIndex == 0:
        bestMove = max(results)
        #print bestMove
        return bestMove
    else:
        bestMove = min(results)
        #print bestMove
        return bestMove


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return self.AlphaBeta(gameState, 1, 0, -sys.maxint, sys.maxint)

  def AlphaBeta(self, gameState, currentDepth, agentIndex, alpha, beta):
    "terminal check"
    if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    
    "alpha-beta algorithm"
    legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
    
    # update next depth
    nextIndex = agentIndex + 1
    nextDepth = currentDepth
    if nextIndex >= gameState.getNumAgents():
        nextIndex = 0
        nextDepth += 1
    
    if agentIndex == 0 and currentDepth == 1: # pacman first move
        results = [self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta) for action in legalMoves]
        bestMove = max(results)
        bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print 'pacman %d' % bestMove
        return legalMoves[chosenIndex]
    
    if agentIndex == 0:
        bestMove = -sys.maxint
        for action in legalMoves:
            bestMove = max(bestMove,\
                           self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta))
            if bestMove >= beta:
                return bestMove
            alpha = max(alpha, bestMove)
        return bestMove
    else:
        bestMove = sys.maxint
        for action in legalMoves:
            bestMove = min(bestMove,\
                           self.AlphaBeta( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex, alpha, beta))
            if alpha >= bestMove:
                return bestMove
            beta = min(beta, bestMove)
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    return self.ExpectiMax(gameState, 1, 0)

  def ExpectiMax(self, gameState, currentDepth, agentIndex):
    "terminal check"
    if currentDepth > self.depth or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    
    "expectimax algorithm"
    legalMoves = [action for action in gameState.getLegalActions(agentIndex) if action!='Stop']
    
    # update next depth
    nextIndex = agentIndex + 1
    nextDepth = currentDepth
    if nextIndex >= gameState.getNumAgents():
        nextIndex = 0
        nextDepth += 1
    
    results = [self.ExpectiMax( gameState.generateSuccessor(agentIndex, action) , nextDepth, nextIndex) for action in legalMoves]
        
    if agentIndex == 0 and currentDepth == 1: # pacman first move
        bestMove = max(results)
        bestIndices = [index for index in range(len(results)) if results[index] == bestMove]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print 'pacman %d' % bestMove
        return legalMoves[chosenIndex]
    
    if agentIndex == 0:
        bestMove = max(results)
        #print bestMove
        return bestMove
    else:
        "In ghost node, return the average(expected) value of action"
        bestMove = sum(results)/len(results)
        #print bestMove, sum(results), len(results)
        return bestMove

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:  see the description document, please
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    ""
    if currentGameState.isWin() :  return sys.maxint
    if currentGameState.isLose() :  return -sys.maxint
    
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    capsulePos = currentGameState.getCapsules()
    
    weightFood, weightGhost, weightCapsule, weightHunter = 5.0, 5.0, 5.0, 0.0
    ghostScore, capsuleScore, hunterScore = 0.0, 0.0, 0.0
    #print '\nnewPos '+str(newPos)
    #print 'oldFood '+str(oldFood.asList())
    #print "score %d" % successorGameState.getScore()
    
    "obtain food score" # may closestFood be zero?
    currentFoodList = currentFood.asList()
    closestFood = min([util.manhattanDistance(currentPos, foodPos) for foodPos in currentFoodList])
    foodScore = 1.0 / closestFood
    
    "obtain ghost, capsule, hunting score"
    if GhostStates:
        ghostPositions = [ghostState.getPosition() for ghostState in GhostStates]
        ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
        ghostDistances = [util.manhattanDistance(currentPos, ghostPos) for ghostPos in ghostPositions]
        
        if sum(ScaredTimes) == 0 : # escape and eat mode
            closestGhost = min(ghostDistances)
            ghostCenterPos = ( sum([ghostPos[0] for ghostPos in ghostPositions])/len(GhostStates),\
                               sum([ghostPos[1] for ghostPos in ghostPositions])/len(GhostStates))
            ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
            #print 'center ' + str(ghostCenterPos)
            if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 20.0, (1.0 / closestCapsule)
                        weightGhost, ghostScore = 3.0, (-1.0 / (ghostCenterDist+1))
                    else:
                        weightGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
                else:
                    weightGhost, ghostScore = 10.0, (-1.0 / (ghostCenterDist+1))
            elif ghostCenterDist >= closestGhost and closestGhost >= 1 :
                weightFood *= 2
                if len(capsulePos) != 0:
                    closestCapsule = min([util.manhattanDistance(capsule,currentPos) for capsule in capsulePos])
                    if closestCapsule <= 3:
                        weightCapsule, capsuleScore = 15.0, (1.0 / closestCapsule)
                        weightGhost, ghostScore = 3.0, (-1.0 / closestGhost)
                    else:
                        ghostScore = -1.0 / closestGhost
                else:
                    ghostScore = -1.0 / closestGhost
            elif closestGhost == 0:
                return -sys.maxint
            elif closestGhost == 1:
                weightGhost, ghostScore = 15.0, (-1.0 / closestGhost)
            else:
                ghostScore = -1.0 / closestGhost
        else: # hunter mode
            normalGhostDist = []
            closestPrey = sys.maxint
            ghostCenterX, ghostCenterY = 0.0, 0.0
            for (index, ghostDist) in enumerate(ghostDistances):
                if ScaredTimes[index] == 0 :
                    normalGhostDist.append(ghostDist)
                    ghostCenterX += ghostPositions[index][0]
                    ghostCenterY += ghostPositions[index][1]
                else:
                    if ghostDist <= ScaredTimes[index] :
                        if ghostDist < closestPrey:
                            closestPrey = ghostDistances[index]
            if normalGhostDist:
                closestGhost = min(normalGhostDist)
                ghostCenterPos = ( ghostCenterX/len(normalGhostDist), ghostCenterY/len(normalGhostDist))
                ghostCenterDist = util.manhattanDistance(currentPos, ghostCenterPos)
                if ghostCenterDist <= closestGhost and closestGhost >= 1 and closestGhost <= 5:
                    weightGhost, ghostScore = 10.0, (- 1.0 / (ghostCenterDist+1))
                elif ghostCenterDist >= closestGhost and closestGhost >= 1 :
                    ghostScore = -1.0 / closestGhost
                elif closestGhost == 0:
                    return -sys.maxint
                elif closestGhost == 1:
                    weightGhost, ghostScore = 15.0, (-1.0 / closestGhost)
                else:
                    ghostScore = - 1.0 / closestGhost
            weightHunter, hunterScore = 35.0, (1.0 / closestPrey)
    
    "a new evaluation function."
    heuristic = currentGameState.getScore() + \
                weightFood*foodScore + weightGhost*ghostScore + \
                weightCapsule*capsuleScore + weightHunter*hunterScore
    return heuristic


# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

