
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

class ImmuneCellType(Enum):
    HELPER_CELL = 0
    B_CELL = 1
    T_CELL = 2
    
class Direction(Enum):
    CENTER = 0
    LEFT = 1
    RIGHT = 2 
    UP = 3
    DOWN = 4

class ImmuneCell:
    
    def __init__(self, x, y, cellType, inflammationRate, antigenAffinity):
        self.x = x
        self.y = y
        self.cellType = cellType
        self.inflammationRate = inflammationRate
        self.antigenAffinity = antigenAffinity
        self.delete = False
        
        #Chemotaxis parameters
        self.D = 0.1
        self.chi0 = 0.2
        self.alpha = 0.6
        self.k = 0.5
        
        self.stepDistance = 2
        self.stepInDir = np.sqrt(self.stepDistance/2)
        
        
        #Activation parameter
        if(self.cellType == ImmuneCellType.T_CELL):
            self.active = False
        else:
            self.active = True
            
        self.life = 0
    
    #c is the cytokin concentration
    def moveCell(self, automatonWidth, automatonHeight, c):
        
        #In this version of the model only T-Cells will move. 
        if(self.cellType == ImmuneCellType.T_CELL):
            if(self.active == False):
                direction = self.getDirectionWithProbabilities([0,1/4,1/4,1/4,1/4])
            else:
                i = self.y
                j = self.x
                
                #p0 = 1 - (4*self.k*self.D) + (self.k*self.alpha*self.chi(c[i,j]))/(4*(1 + self.alpha*c[i,j]))*((c[i, (j+1)%automatonWidth] - c[i, (j-1)%automatonWidth])**2 + (c[(i-1)%automatonHeight,j] - c[(i+1)%automatonHeight,j])**2)-(self.k*self.chi(c[i,j]))*(c[(i+1)%automatonHeight,j] + c[(i-1)%automatonHeight,j] + c[i,(j+1)%automatonWidth] + c[i,(j-1)%automatonWidth] -4*c[i,j])
                p0 = 0
                p1 = (self.k*self.D) - (self.k/4)*(self.chi(c[i,j])*(c[i,(j+1)%automatonWidth] - c[i,(j-1)%automatonWidth]))
                p2 = (self.k*self.D) + (self.k/4)*(self.chi(c[i,j])*(c[i,(j+1)%automatonWidth] - c[i,(j-1)%automatonWidth]))
                p3 = (self.k*self.D) + (self.k/4)*(self.chi(c[i,j])*(c[(i-1)%automatonHeight,j] - c[(i+1)%automatonHeight,j]))
                p4 = (self.k*self.D) - (self.k/4)*(self.chi(c[i,j])*(c[(i-1)%automatonHeight,j] - c[(i+1)%automatonHeight,j]))
                
                probabilities = self.normalizeProbabilities([p0,p1,p2,p3,p4])
                #print(probabilities)
                direction = self.getDirectionWithProbabilities(probabilities)
                
            
            if(direction == Direction.CENTER):
                self.x = self.x
                self.y = self.y
            elif(direction == Direction.UP):
                #y-axis is inverted
                self.y = int((self.y - self.stepInDir)%automatonHeight)
            elif(direction == Direction.DOWN):
                self.y = int((self.y + self.stepInDir)%automatonHeight)
            elif(direction == Direction.LEFT):
                self.x = int((self.x - self.stepInDir)%automatonWidth)
            elif(direction == Direction.RIGHT):
                self.x = int((self.x + self.stepInDir)%automatonWidth)
    
    def getDirectionWithProbabilities(self, probabilities):
        accumulatedProbability = 0
        randomVal = random.random()
        for i in range(0,len(probabilities)):
            if(randomVal > accumulatedProbability and randomVal <= accumulatedProbability + probabilities[i]):
                return Direction(i)
            
            accumulatedProbability = accumulatedProbability + probabilities[i]
        
        return -1
    
    def normalizeProbabilities(self, pValues):
        M = np.max(pValues)
        m = np.min(pValues)
        normalizedProbabilities = []
        for i in range(0,len(pValues)):
            normalizedProbability = (pValues[i] - m)/(M - m)
            normalizedProbabilities.append(normalizedProbability)
        normalizedProbabilities = np.array(normalizedProbabilities)/sum(normalizedProbabilities)
        return normalizedProbabilities
    
    def chi(self, c):
        return self.chi0/(1 + self.alpha*c)
    
    def isActive(self):
        return self.active
    
    def setDelete(self, delete):
        self.delete = delete
    
    def isDeleted(self):
        return self.delete
    
    def setActive(self, active):
        self.active = active
    
    def getLife(self):
        return self.life
    
    def setLife(self, life):
        self.life = life
            
                        
        
        

class ImmuneAutomaton:
    
    def __init__(self, automatonWidth, automatonHeight):
        self.automatonWidth = automatonWidth
        self.automatonHeight = automatonHeight
        
        #Cytokine production and dissipation parameters
        self.bCellInflammation = 0.05
        self.tCellInflammation = 0.005
        self.helperCellInflammation = 0.05
        self.cytokineDissipation = 0.1
        self.cytokineDiffusion = 0.5
        
        #Probability of attack by helper cells and b-cells
        self.rHelper = 0.001
        self.rBCell = 0.1
        self.rAntibody = 1
        self.rTAttack = 1
        
        #Threshold of suppressor that makes the antigen undetectable 
        self.Tsupp = 0.5
         
        self.minTCellProductionRate = 1
        self.maxTCellProductionRate = 20
        self.tCellProductionRate = 1
        
        self.maxNTCells = 300
        
        #Healthy case
        self.antigenAffinity = 1
        
        self.maxTCellLife = 1000
        
        self.initializeAutomaton()
        
        self.evasionProbability = 0
        
        self.attackedPositions = []
        
        self.capCytokineConcentration = self.bCellInflammation + self.helperCellInflammation + self.tCellInflammation
        
    def updateCapCytokine(self):
        self.capCytokineConcentration = self.bCellInflammation + self.helperCellInflammation + self.tCellInflammation
    
    def setEvasionProbability(self, evasionProbability):
        self.evasionProbability = evasionProbability
    
        
    def isAntigenTrapped(self, index1, index2):
        bCellIndex = self.getBCellIndexAtPosition(index1, index2)
        if(bCellIndex == -1 and not [index1, index2] in self.attackedPositions):
            return False
        return True
    
    def wasDisposed(self, i,j):
        return [i,j] in self.attackedPositions
    
    def setAntigenPositions(self, antigenPositions):
        self.antigenPositions = antigenPositions
        
    def activateImmuneDisease(self):
        self.antigenAffinity = 0.3
        self.rTAttack = 0.3
    
    def getActiveTCellNumber(self):
        activeTCells = 0
        for i in range(0,len(self.tCells)):
            if(self.tCells[i].isActive()):
                activeTCells = activeTCells + 1
                
        return activeTCells
    
    def resetAttackedPositions(self):
        self.attackedPositions = []
    
    def addCytokine(self, quantity,i,j):
        if(self.cytokineConcentration[i,j] +quantity <= self.capCytokineConcentration and self.cytokineConcentration[i,j] + quantity > 0 ):
            self.cytokineConcentration[i,j] = self.cytokineConcentration[i,j] + quantity
        #elif(self.cytokineConcentration[i,j] +quantity >=self.capCytokineConcentration):
            #print("Exceeded "+str(quantity))
            #print(self.cytokineDissipation)
    
    def stepImmuneAutomaton(self):
        #Update the T-Cell Rate depending on the number of active T-Cells
        activeTCells = self.getActiveTCellNumber()
        self.tCellProductionRate = int(self.minTCellProductionRate + (activeTCells/self.maxTCellProductionRate)*(self.maxTCellProductionRate - self.minTCellProductionRate))
        
        #Add new random new cells in the border of the automaton
        if(len(self.tCells) < self.maxNTCells):
            for i in range(0,self.tCellProductionRate):
                #Select a random boundary 
                boundaryIndex = random.randint(0,3)
                #Upper boundary
                if(boundaryIndex == 0):
                    randPos = random.randint(0,self.automatonWidth-1)
                    self.addTCell(0,randPos)
                #Bottom boundary
                elif(boundaryIndex == 1):
                    randPos = random.randint(0,self.automatonWidth-1)
                    self.addTCell(self.automatonHeight-1, randPos)
                #Left boundary
                elif(boundaryIndex == 2):
                    randPos = random.randint(0,self.automatonHeight-1)
                    self.addTCell(randPos, 0)
                #Right boundary
                elif(boundaryIndex == 3):
                    randPos = random.randint(0,self.automatonHeight-1)
                    self.addTCell(randPos, self.automatonWidth-1)
        
        #Update H-Cells, B-Cells, and antibodies
        self.updateHCells()
        self.updateBCells()
        self.updateAntibodies()
        self.updateAntigenEvasion()
        
        #Move T-Cells and attack antibodies
        for i in range(0,len(self.tCells)):
            self.tCells[i].moveCell(self.automatonWidth, self.automatonHeight, self.cytokineConcentration)
            index1 = self.tCells[i].y
            index2 = self.tCells[i].x
            if(self.antibodyGrid[index1, index2] == 1 and random.random()< self.rTAttack):
                #Kill the antigen there
                self.antigenPositions[index1, index2] = 0
                self.antibodyGrid[index1, index2] = 0
                bCellIndex = self.getBCellIndexAtPosition(index1, index2)
                hCellIndex = self.getHCellIndexAtPosition(index1, index2)
                self.attackedPositions.append([index1, index2])
                self.addCytokine(self.tCellInflammation, index1, index2)
                #print(len(self.attackedPositions))
                if(bCellIndex != -1):
                    self.bCells[bCellIndex].setDelete(True)
                else:
                    print("-1 found")
                if(hCellIndex != -1):
                    self.helperCells[hCellIndex].setDelete(True)
                else:
                    print("-1 found")
                
                if(not self.tCells[i].isActive()):
                    self.tCells[i].setActive(True)
            
            #Autoimmune effect. The parameter antigenAffinity can be taken as an indicator of how sick the system is
            elif(random.random() < 1 - self.antigenAffinity and not (index1 == 0 or index2 == 0)):
                self.addCytokine(self.tCellInflammation, index1, index2)
                if(not self.tCells[i].isActive()):
                    self.tCells[i].setActive(True)
            
            self.tCells[i].setLife(self.tCells[i].getLife() + 1)
        
            #Decide if the T-Cell will die of old 
            if(self.tCells[i].getLife() > self.maxTCellLife):
                self.tCells[i].setDelete(True)
        
        #Eliminate cells that are scheduled to be removed
        self.removeEliminatedCells()
        self.spawnCytokines()
        #Diffuse cytokine 
        self.diffuseCytokines()
        
    def removeEliminatedCells(self):
        bCellsToRemove = []
        tCellsToRemove = []
        helperCellsToRemove = []
        for i in range(0,len(self.bCells)):
            if(self.bCells[i].isDeleted()):
                bCellsToRemove.append(self.bCells[i])
        
        for i in range(0,len(self.tCells)):
            if(self.tCells[i].isDeleted()):
                tCellsToRemove.append(self.tCells[i])
        
        for i in range(0,len(self.helperCells)):
            if(self.helperCells[i].isDeleted()):
                helperCellsToRemove.append(self.helperCells[i])
        
        for i in range(0,len(bCellsToRemove)):
            self.bCells.remove(bCellsToRemove[i])
        
        for i in range(0,len(helperCellsToRemove)):
            self.helperCells.remove(helperCellsToRemove[i])
                
        for i in range(0,len(tCellsToRemove)):
            self.tCells.remove(tCellsToRemove[i])
    
    
    def getBCellIndexAtPosition(self,i,j):
        for s in range(0,len(self.bCells)):
            if(self.bCells[s].y == i and self.bCells[s].x == j):
                return s
        
        return -1
    
    def getHCellIndexAtPosition(self, i, j):
        for s in range(0,len(self.helperCells)):
            if(self.helperCells[s].y == i and self.helperCells[s].x == j):
                return s
        return -1
                
    
    def isHCellAtPosition(self,i,j):
        for s in range(0,len(self.helperCells)):
            if(self.helperCells[s].y == i and self.helperCells[s].x == j):
                return True
        
        return False
    
    def isBCellAtPosition(self, i, j):
        for s in range(0,len(self.bCells)):
            if(self.bCells[s].y == i and self.bCells[s].x == j):
                return True
        
        return False
    
    def updateHCells(self):
        n = np.size(self.antigenPositions,0)
        m = np.size(self.antigenPositions,1)

        for i in range(0,n):
            for j in range(0,m):
                if(self.antigenPositions[i,j] == 1):
                    randVal = random.random()
                    if(randVal < self.rHelper):
                        if(not self.isHCellAtPosition(i,j)):
                            self.addHCell(i,j)
                            self.addCytokine(self.helperCellInflammation, i,j)
    
    def updateBCells(self):
        for i in range(0,len(self.helperCells)):
            randVal = random.random()
            index1 = self.helperCells[i].y
            index2 = self.helperCells[i].x
            
            if(randVal < self.rBCell and self.antigenPositions[index1,index2] == 1):
                index1 = self.helperCells[i].y
                index2 = self.helperCells[i].x
                if(not self.isBCellAtPosition(index1, index2)):
                    self.addBCell(index1, index2)
                    self.addCytokine(self.bCellInflammation, index1, index2)
            
            if(self.antigenPositions[index1, index2] == 0):
                self.helperCells[i].setDelete(True)
    
    def updateAntigenEvasion(self):
        for i in range(0,len(self.bCells)):
            if(random.random() < self.evasionProbability):
                self.bCells[i].setDelete(True)
                index1 = self.bCells[i].y
                index2 = self.bCells[i].x
                helperCellIndex = self.getHCellIndexAtPosition(index1, index2)
                if(helperCellIndex != -1):
                    self.helperCells[helperCellIndex].setDelete(True)
                else:
                    print("Something went wrong with the helper cell index")
                if(self.antibodyGrid[index1, index2] == 1):
                    self.antibodyGrid[index1, index2] = 0
    
    
        
    def updateAntibodies(self):
        for i in range(0,len(self.bCells)):
            randVal = random.random()
            index1 = self.bCells[i].y
            index2 = self.bCells[i].x
            if(self.antigenPositions[index1, index2] == 1):
                if(randVal < self.rAntibody and not self.antibodyGrid[index1, index2] == 1):
                    self.antibodyGrid[index1, index2] = 1
            elif(self.antigenPositions[index1, index2] == 0):
                self.bCells[i].setDelete(True)
                helperCellIndex = self.getHCellIndexAtPosition(index1, index2)
                if(helperCellIndex != -1):
                    self.helperCells[helperCellIndex].setDelete(True)
                else:
                    print("Incorrect index for helper cell")
                
                if(self.antibodyGrid[index1, index2] == 1):
                    self.antibodyGrid[index1, index2] = 0
            
                
    def spawnCytokines(self):
        for i in range(0,len(self.bCells)):
            index1 = self.bCells[i].y
            index2 = self.bCells[i].x
            if(self.antibodyGrid[index1,index2] == 1):
                self.cytokineConcentration[index1, index2] = self.helperCellInflammation + self.bCellInflammation
        
    
    def diffuseCytokines(self):
        n = np.size(self.cytokineConcentration,0)
        m = np.size(self.cytokineConcentration,1)
        previousCytokines = self.cytokineConcentration.copy()
        for i in range(0,n):
            for j in range(0,m):
                if(self.antibodyGrid[i,j] == 0):
                    delta = self.cytokineDiffusion*(previousCytokines[(i+1)%self.automatonHeight,j] + previousCytokines[(i-1)%self.automatonHeight,j] + previousCytokines[i,(j-1)%self.automatonWidth] + previousCytokines[i,(j+1)%self.automatonWidth] - 4*previousCytokines[i,j]) - self.cytokineDissipation
                    self.addCytokine(delta,i,j)
    
    
    def getPicture(self):
        antigenColor = [255/255, 221/255, 84/255]
        tCellColorInactive = [222/255, 29/255, 29/255]
        tCellColorActive = [101/255, 255/255, 18/255]
        antibodyColor = [23/255, 152/255, 232/255]
        
        picture = np.ones((self.automatonHeight, self.automatonWidth, 3))
        
        for i in range(0,self.automatonHeight):
            for j in range(0, self.automatonWidth):
                if(self.antigenPositions[i,j] == 1):
                    picture[i,j,:] = antigenColor
                
                if(self.antibodyGrid[i,j] == 1):
                    picture[i,j,:] = antibodyColor
                
        for i in range(0,len(self.tCells)):
            tCell = self.tCells[i]
            if(tCell.isActive()):
                picture[tCell.y, tCell.x,:] = tCellColorActive
            else:
                picture[tCell.y, tCell.x,:] = tCellColorInactive
                
            
        
        return picture
    
    
    def evolveWithMovie(self,nSteps):
        movie = np.zeros((self.automatonHeight, self.automatonWidth, 3, nSteps + 1))
        movie[:,:,:,0] = self.getPicture()
        for i in tqdm(range(0,nSteps)):
            self.stepImmuneAutomaton()
            movie[:,:,:,i+1] = self.getPicture()
        
        return movie
    
    
    def addHCell(self, i, j):
        hCell = ImmuneCell(j,i,ImmuneCellType.HELPER_CELL, self.helperCellInflammation, self.antigenAffinity)
        self.helperCells.append(hCell)
        
    def addBCell(self, i, j):
        bCell = ImmuneCell(j,i,ImmuneCellType.B_CELL, self.bCellInflammation, self.antigenAffinity)
        self.bCells.append(bCell)
    
    def addTCell(self, i, j):
        tCell = ImmuneCell(j,i, ImmuneCellType.T_CELL, self.tCellInflammation, self.antigenAffinity)
        self.tCells.append(tCell)
        
    def setAntigenPositions(self,antigenPositions):
        self.antigenPositions = antigenPositions
        
    
    def initializeAutomaton(self):
        self.helperCells = []
        self.bCells = []
        self.tCells = []
        self.antibodyGrid = np.zeros((self.automatonHeight, self.automatonWidth))
        self.cytokineConcentration = np.zeros((self.automatonHeight, self.automatonWidth))
        self.suppressorConcentration = np.zeros((self.automatonHeight, self.automatonWidth))
        self.antigenPositions = np.zeros((self.automatonHeight, self.automatonWidth))
        
    
    
        
        
    