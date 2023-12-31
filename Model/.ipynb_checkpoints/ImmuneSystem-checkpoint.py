
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

class AntigenType(Enum):
    #Mechanisms for different antigen types differ in the immune system
    CELL = 0
    EXTERNAL = 1
    

class ImmuneCell:
    
    def __init__(self, x, y, cellType, inflammationRate, antigenAffinity):
        self.x = x
        self.y = y
        self.cellType = cellType
        self.inflammationRate = inflammationRate
        self.antigenAffinity = antigenAffinity
        self.delete = False
        
        #Chemotaxis parameters
        self.D = 0.4
        self.chi0 = 0.35
        self.alpha = 0.1
        self.k = 0.5
        
        self.stepDistance = 2
        self.stepInDir = np.sqrt(self.stepDistance/2)
        
        
        #Activation parameter
        if(self.cellType == ImmuneCellType.T_CELL):
            self.active = False
        else:
            self.active = True
            
        self.life = 0
    
    def __eq__(self,other):
        return (self.x == other.x) and (self.y == other.y) and (self.cellType == other.cellType)
    
    #c is the cytokin concentration
    def moveCell(self, automatonWidth, automatonHeight, c):
        
        #In this version of the model only Killer cells will move. 
        if(self.cellType == ImmuneCellType.T_CELL):
            if(self.active == False):
                direction = self.getDirectionWithProbabilities([0,1/4,1/4,1/4,1/4])
            else:
                i = self.y
                j = self.x
                p0 = 0
                p1 = (self.k*self.D) - (self.k/4)*(self.chi(c[i,j])*(c[i,(j+1)%automatonWidth] - c[i,(j-1)%automatonWidth]))
                p2 = (self.k*self.D) + (self.k/4)*(self.chi(c[i,j])*(c[i,(j+1)%automatonWidth] - c[i,(j-1)%automatonWidth]))
                p3 = (self.k*self.D) + (self.k/4)*(self.chi(c[i,j])*(c[(i-1)%automatonHeight,j] - c[(i+1)%automatonHeight,j]))
                p4 = (self.k*self.D) - (self.k/4)*(self.chi(c[i,j])*(c[(i-1)%automatonHeight,j] - c[(i+1)%automatonHeight,j]))
                
                probabilities = self.normalizeProbabilities([p0,p1,p2,p3,p4])
                #print(probabilities)
                direction = self.getDirectionWithProbabilities(probabilities)
                
                #print(direction)
                #print(c[(i-1)%automatonHeight,j]-c[i,j])
                #print(c[(i+1)%automatonHeight,j]-c[i,j])
                #print(c[i,(j+1)%automatonWidth]-c[i,j])
                #print(c[i,(j-1)%automatonWidth]-c[i,j])
                
            
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
            if(pValues[i] == 0):
                normalizedProbabilities.append(0)
            else:
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
    
    def __init__(self, automatonWidth, automatonHeight, antigenType):
        self.antigenType = antigenType
        self.automatonWidth = automatonWidth
        self.automatonHeight = automatonHeight
        
        #Cytokine production and dissipation parameters
        self.bCellInflammation = 0.5
        self.tCellInflammation = 1.5
        self.helperCellInflammation = 0.5
        self.cytokineDissipation = 0.02
        self.cytokineDiffusion = 0.1
        
        self.rHelper = 0.1
        self.rBCell = 0.1
        self.rTAttack = 1
        self.rAntibody = 1
         
        self.minTCellProductionRate = 1
        self.maxTCellProductionRate = 20
        self.tCellProductionRate = 1
        
        self.maxNTCells = 400
        
        #Healthy case
        self.antigenAffinity = 1
        self.maxTCellLife = 3000
        self.initializeAutomaton()
        self.evasionProbability = 0
        if(self.antigenType == AntigenType.CELL):
            self.evasionProbability = 0.01
        
        self.attackedPositions = []
        self.tCellAutoimmuneInflammation = 0.25
        self.eliminationProb = 0
        self.boundarySpawn = True
        
    def updateCapCytokine(self):
        self.capCytokineConcentration = self.bCellInflammation + self.helperCellInflammation + self.tCellInflammation
    
    def setEvasionProbability(self, evasionProbability):
        self.evasionProbability = evasionProbability
    
    def isAntigenTrapped(self, index1, index2):
        if(self.antigenType == AntigenType.EXTERNAL):
            if(self.isBCellAtPosition(index1, index2) and self.antibodyGrid[index1, index2] == 1):
                return True
            else:
                return False
        
        elif(self.antigenType == AntigenType.CELL):
            if(self.isHCellAtPosition(index1, index2) and self.antigenPositions[index1, index2] == 1):
                return True
            else:
                return False
    
    def wasDisposed(self, i,j):
        return [i,j] in self.attackedPositions
    
    def setAntigenPositions(self, antigenPositions):
        self.antigenPositions = antigenPositions
        
    def activateImmuneDisease(self):
        if(self.antigenType == AntigenType.EXTERNAL):
            self.antigenAffinity = 0.5
            self.rAntibody = 0.1
        elif(self.antigenType == AntigenType.CELL):
            self.antigenAffinity = 0.5
            
    
    def activateImmuneDiseaseWithValues(self, antigenAffinity, rAntibody):
        if(self.antigenType == AntigenType.EXTERNAL):
            self.antigenAffinity = antigenAffinity
            self.rAntibody = rAntibody
        
        elif(self.antigenType== AntigenType.CELL):
            self.antigenAffinity = antigenAffinity
    
    def getActiveTCellNumber(self):
        activeTCells = 0
        for i in range(0,len(self.tCells)):
            if(self.tCells[i].isActive()):
                activeTCells = activeTCells + 1
                
        return activeTCells
    
    def resetAttackedPositions(self):
        self.attackedPositions = []
    
    def addCytokine(self, quantity,i,j):
        self.cytokineConcentration[i,j] = self.cytokineConcentration[i,j] + quantity
            
    def spawnNewTCell(self, inBoundary):
        
        if(inBoundary):
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
        else:
            index1 = random.randint(0,self.automatonHeight-1)
            index2 = random.randint(0,self.automatonWidth-1)
            
            self.addTCell(index1, index2)
    
    def attackCondition(self,i,j):
        if(self.antigenType == AntigenType.EXTERNAL):
            return self.antibodyGrid[i,j] == 1
        elif(self.antigenType == AntigenType.CELL):
            return self.helperCellPositions[i,j] == 1
        
    def stepImmuneAutomaton(self):
        #print(len(self.helperCells))
        #Update the T-Cell Rate depending on the number of active T-Cells
        activeTCells = self.getActiveTCellNumber()
        self.tCellProductionRate = int(self.minTCellProductionRate + (activeTCells/self.maxTCellProductionRate)*(self.maxTCellProductionRate - self.minTCellProductionRate))
        
        #Add new random new cells in the border of the automaton
        if(len(self.tCells) < self.maxNTCells):
            for i in range(0,self.tCellProductionRate):
                self.spawnNewTCell(self.boundarySpawn)
        
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
            if(self.attackCondition(index1, index2) and random.random()< self.rTAttack):
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
                if(hCellIndex != -1):
                    self.helperCells[hCellIndex].setDelete(True)
                
                self.helperCellPositions[index1, index2] = 0
                
                if(not self.tCells[i].isActive()):
                    self.tCells[i].setActive(True)
            
            #Autoimmune effect. The parameter antigenAffinity can be taken as an indicator of how sick the system is
            elif(random.random() < 1 - self.antigenAffinity and random.random() < self.rTAttack):
                self.addCytokine(self.tCellAutoimmuneInflammation, index1, index2)
                if(not self.tCells[i].isActive()):
                    self.tCells[i].setActive(True)
            
            self.tCells[i].setLife(self.tCells[i].getLife() + 1)
        
            #Decide if the T-Cell will die of old 
            if(self.tCells[i].getLife() > self.maxTCellLife):
                self.tCells[i].setDelete(True)
        
        self.updateSuppressionEffect()
        
        #Eliminate cells that are scheduled to be removed
        self.removeEliminatedCells()
        self.spawnCytokines()
        #Diffuse cytokine 
        self.diffuseCytokines()
    
    def updateSuppressionEffect(self):
        nCells = len(self.tCells)
        if(nCells > self.maxNTCells):
            for i in range(0,nCells):
                if(random.random() < self.eliminationProb):
                    self.tCells[i].setDelete(True)
        
        
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
        cell = ImmuneCell(j,i,ImmuneCellType.HELPER_CELL,0,0)
        if(cell in self.helperCells):
            return True
        return False
    
    def isBCellAtPosition(self, i, j):
        cell = ImmuneCell(j,i,ImmuneCellType.B_CELL,0,0)
        if(cell in self.bCells):
            return True
        return False
    
    def updateHCells(self):
        n = np.size(self.antigenPositions,0)
        m = np.size(self.antigenPositions,1)
        
        if(self.antigenType == AntigenType.EXTERNAL):
            for i in range(0,n):
                for j in range(0,m):
                    if(self.antigenPositions[i,j] == 1):
                        randVal = random.random()
                        if(randVal < self.rHelper):
                            if(not self.isHCellAtPosition(i,j)):
                                self.addHCell(i,j)
                                self.addCytokine(self.helperCellInflammation, i,j)
        elif(self.antigenType == AntigenType.CELL):
            for i in range(0,n):
                for j in range(0,m):
                    if(self.antigenPositions[i,j] == 1):
                        randVal = random.random()
                        if(randVal < self.rHelper and self.helperCellPositions[i,j] == 0):
                            self.addHCell(i,j)
                            self.addCytokine(self.helperCellInflammation, i,j)
                            self.helperCellPositions[i,j] = 1
                    elif(self.antigenPositions[i,j] == 0 and self.helperCellPositions[i,j] == 1):
                        self.helperCellPositions[i,j] = 0
                        indexCell = self.getHCellIndexAtPosition(i,j)
                        if(indexCell != -1):
                            self.helperCells[indexCell].setDelete(True)
                    
            
        
    
    def updateBCells(self):
        if(self.antigenType == AntigenType.EXTERNAL):
            for i in range(0,len(self.helperCells)):
                randVal = random.random()
                index1 = self.helperCells[i].y
                index2 = self.helperCells[i].x
                
                if(randVal < self.rBCell):
                    index1 = self.helperCells[i].y
                    index2 = self.helperCells[i].x
                    
                    if(not self.isBCellAtPosition(index1, index2)):
                        self.addBCell(index1, index2)
                        self.addCytokine(self.bCellInflammation, index1, index2)
    
    def updateAntigenEvasion(self):
        if(self.antigenType == AntigenType.EXTERNAL):
        
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
        
        elif(self.antigenType == AntigenType.CELL):
            
            for i in range(0,len(self.helperCells)):
                if(random.random() < self.evasionProbability):
                    self.helperCells[i].setDelete(True)
    
    def updateAntibodies(self):
        if(self.antigenType == AntigenType.EXTERNAL):
            for i in range(0,len(self.bCells)):
                randVal = random.random()
                index1 = self.bCells[i].y
                index2 = self.bCells[i].x
                if(randVal < self.rAntibody and not self.antibodyGrid[index1, index2] == 1):
                    self.antibodyGrid[index1, index2] = 1
                
    def spawnCytokines(self):
        if(self.antigenType == AntigenType.EXTERNAL):
            for i in range(0,len(self.bCells)):
                index1 = self.bCells[i].y
                index2 = self.bCells[i].x
                if(self.antibodyGrid[index1,index2] == 1):
                    self.cytokineConcentration[index1, index2] = self.helperCellInflammation + self.bCellInflammation

        elif(self.antigenType == AntigenType.CELL):
            for i in range(0,len(self.helperCells)):
                index1 = self.helperCells[i].y
                index2 = self.helperCells[i].x
                
                self.cytokineConcentration[index1, index2] = self.helperCellInflammation
        
    
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
        helperCellColor = [134/255, 52/255, 235/255]
        
        picture = np.ones((self.automatonHeight, self.automatonWidth, 3))
        
        for i in range(0,self.automatonHeight):
            for j in range(0, self.automatonWidth):
                if(self.antigenPositions[i,j] == 1):
                    picture[i,j,:] = antigenColor
                
                if(self.antigenType == AntigenType.EXTERNAL):
                    if(self.antibodyGrid[i,j] == 1):
                        picture[i,j,:] = antibodyColor
                elif(self.antigenType == AntigenType.CELL):
                    if(self.isHCellAtPosition(i,j)):
                        picture[i,j,:] = helperCellColor
                
        for i in range(0,len(self.tCells)):
            tCell = self.tCells[i]
            if(tCell.isActive()):
                picture[tCell.y, tCell.x,:] = tCellColorActive
            else:
                picture[tCell.y, tCell.x,:] = tCellColorInactive
                
            
        
        return picture
    
    def getCellCounts(self):
        antigenCount = sum(sum(self.antigenPositions))
        hCounts = len(self.helperCells)
        bCounts = len(self.bCells)
        antibodyCounts = sum(sum(self.antibodyGrid))
        activeKillerCells = 0
        
        for i in range(0,len(self.tCells)):
            cell = self.tCells[i]
            if(cell.isActive()):
                activeKillerCells = activeKillerCells + 1
        
        return np.array([antigenCount, hCounts, bCounts, antibodyCounts, activeKillerCells])
    
    def getRefinedPicture(self):
        picture = np.ones((self.automatonHeight, self.automatonWidth, 3))
        if(self.antigenType == AntigenType.EXTERNAL):
            antigenColor = [63/255, 237/255, 47/255]
            antibodyColor = [46/255, 166/255, 240/255]
            helperCellColor = [255/255, 153/255, 0/255]
            bCellColor = [133/255, 103/255, 62/255]
            macrophageColor = [255/255, 41/255, 66/255]
            
            
            
            for i in range(0,self.automatonHeight):
                for j in range(0,self.automatonWidth):
                    if(self.antigenPositions[i,j] == 1):
                        picture[i,j,:] = antigenColor
                    
            
            for i in range(0,len(self.helperCells)):
                cell = self.helperCells[i]
                index1 = cell.y
                index2 = cell.x
                picture[index1, index2,:] = helperCellColor
                
            
            for i in range(0,len(self.bCells)):
                cell = self.bCells[i]
                index1 = cell.y
                index2 = cell.x
                picture[index1, index2,:] = bCellColor
            
            
            for i in range(0,self.automatonHeight):
                for j in range(0,self.automatonWidth):
                    if(self.antibodyGrid[i,j] == 1):
                        picture[i,j,:] = antibodyColor
                    
            
            for i in range(0,len(self.tCells)):
                cell = self.tCells[i]
                index1 = cell.y
                index2 = cell.x
                picture[index1, index2, :] = macrophageColor
                
            
            return picture
        
        elif(self.antigenType == AntigenType.CELL):
            
            antigenColor = [63/255, 237/255, 47/255]
            antibodyColor = [46/255, 166/255, 240/255]
            helperCellColor = [255/255, 153/255, 0/255]
            bCellColor = [133/255, 103/255, 62/255]
            macrophageColor = [255/255, 41/255, 66/255]
            
            
            
            for i in range(0,self.automatonHeight):
                for j in range(0,self.automatonWidth):
                    if(self.antigenPositions[i,j] == 1):
                        picture[i,j,:] = antigenColor
                    
            
            for i in range(0,len(self.helperCells)):
                cell = self.helperCells[i]
                index1 = cell.y
                index2 = cell.x
                picture[index1, index2,:] = helperCellColor
                    
            
            for i in range(0,len(self.tCells)):
                cell = self.tCells[i]
                index1 = cell.y
                index2 = cell.x
                picture[index1, index2, :] = macrophageColor
            
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
        self.helperCellPositions = np.zeros((self.automatonHeight, self.automatonWidth))
        self.cytokineConcentration = np.zeros((self.automatonHeight, self.automatonWidth))
        self.suppressorConcentration = np.zeros((self.automatonHeight, self.automatonWidth))
        self.antigenPositions = np.zeros((self.automatonHeight, self.automatonWidth))
        
    
    
        
        
    