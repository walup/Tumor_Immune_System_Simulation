from enum import Enum 
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ImmuneSystem import ImmuneAutomaton

class CellType(Enum):
    PROLIFERATING = [28/255, 241/255, 93/255]
    COMPLEX = [26/255, 69/255, 245/255]
    DEAD = [245/255, 72/255, 27/255]
    NECROTIC = [130/255, 130/255, 130/255]
    
class Cell:
    def __init__(self, x, y, cellType):
        self.x = x
        self.y = y
        self.cellType = cellType
        self.oxygenThreshold = 0.001
        self.quiescent = False
        self.radioAffected = False
        self.chemoAffected = False
        self.countCycle = 0
        
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def turnNecrotic(self):
        self.cellType = CellType.NECROTIC
    
    def breathe(self, oxygenConcentration):
        if(oxygenConcentration < self.oxygenThreshold):
            self.turnNecrotic()
    
    def setQuiescent(self, quiescent):
        self.quiescent = quiescent
    
    def isQuiescent(self):
        return self.quiescent
        
    def isDead(self):
        return self.cellType == CellType.NECROTIC or self.cellType == CellType.DEAD

class Nutrient:
    
    def __init__(self, width, height, diffusionConstant, healthyCellConsumption, consumptionProlif, consumptionQuiescent):
        
        self.width = width
        self.height = height
        self.nutrientConcentration = np.zeros((height, width))
        self.consumptionProlif = consumptionProlif
        self.healthyCellConsumption = healthyCellConsumption
        self.consumptionQuiescent = consumptionQuiescent
        self.diffusionConstant = diffusionConstant
    
    def putValue(self, i,j, value):
        self.nutrientConcentration[i,j] = value
        
    def updateNutrient(self, cell, x, y):
        index1 = y
        index2 = x
        
        if(cell is None):
            
            laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
            deltaConcentration = self.diffusionConstant*laPlacian - self.healthyCellConsumption
            self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
        else:
            if(cell.cellType == CellType.PROLIFERATING and not cell.isQuiescent()):
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionProlif
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            
            elif((cell.cellType == CellType.PROLIFERATING and cell.isQuiescent) or cell.cellType == CellType.COMPLEX):
                
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian - self.consumptionQuiescent
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
            
            else:
                
                laPlacian = self.nutrientConcentration[(index1 + 1)%self.height, index2] + self.nutrientConcentration[(index1 - 1)%self.height, index2] + self.nutrientConcentration[index1, (index2 + 1)%self.width] + self.nutrientConcentration[index1, (index2 - 1)%self.width] - 4*self.nutrientConcentration[index1, index2]
                deltaConcentration = self.diffusionConstant*laPlacian 
                self.nutrientConcentration[index1, index2] = self.nutrientConcentration[index1, index2] + deltaConcentration
                
        
    def getNutrientValue(self, i,j):
        return self.nutrientConcentration[i, j]
    

class ECM:
    def __init__(self, width, height, ec, et):
        self.width = width
        self.height = height
        self.extraCellularMatrix = np.zeros((self.height, self.width))
        self.ec = ec
        self.et = et
        self.initializeMatrix()
        
    def initializeMatrix(self):
        for i in range(1, self.height-1):
            for j in range(1,self.width -1):
                #Initialize randomly with values in the range [0.8,1.2]
                self.extraCellularMatrix[i,j] = 0.8 + random.random()*(1.2 - 0.8)
    
    def updateMatrix(self, nNeighbors,i,j):
        deltaECM = -self.ec*nNeighbors*self.extraCellularMatrix[i,j]
        self.extraCellularMatrix[i,j] = self.extraCellularMatrix[i,j] + deltaECM
    
    def canInvadePosition(self, i, j):
        if(self.extraCellularMatrix[i,j]<self.et):
            return True
        return False
    
class Tissue:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        #Positions occupied by non-necrotic cells
        self.occupiedPositions = np.zeros((self.height, self.width))
        self.necroticPositions = np.zeros((self.height, self.width))
        self.cells = []
        self.colorNecrotic = [176/255, 176/255, 176/255]
        #ECM Params
        self.ec = 0.15
        self.et = 0.05
        #Nutrient params
        self.diffusionConstant = 0.01
        self.consumptionProlif = 0.01
        self.consumptionQuiescent = 0.005
        self.consumptionHealthy = 0.001
        #Proliferation param
        self.rProlif = 0.65
        self.rDecay = 0.4
        self.rProlifPrime = 0.65
        self.K = 1000
        self.inflammationResponseFactor = 0
        
        self.maxProlifImmuneBoost = 2
        self.ratioUpdateImmune = 10
        
        self.initializeNutrientAndECM()
        self.initializeImmuneSystem()
        
        self.rSuppresion = 0
        
        self.therapies = []
    
    
    def initializeNutrientAndECM(self):
        #Create objects for ECM and Nutrient
        self.ecm = ECM(self.width, self.height, self.ec, self.et)
        self.nutrient = Nutrient(self.width, self.height, self.diffusionConstant, self.consumptionHealthy, self.consumptionProlif, self.consumptionQuiescent)
        #Initialize concentrations for ECM
        self.ecm.initializeMatrix()
        for i in range(0,self.height):
            for j in range(0,self.width):
                self.nutrient.nutrientConcentration[i,j] = 1
        
        self.nutrient.nutrientConcentration[0,:] = 2
        self.nutrient.nutrientConcentration[:,0] = 2
        self.nutrient.nutrientConcentration[self.height-1,:] = 2
        self.nutrient.nutrientConcentration[:,self.width-1] = 2
        
    def getTumorRadius(self):
        maxDist = float('-inf')
        xMid = np.floor(self.width/2)
        yMid = np.floor(self.height/2)
        for i in range(0,len(self.cells)):
            cell = self.cells[i]
            if(cell.cellType == CellType.PROLIFERATING):
                xPos = cell.x
                yPos = cell.y
                dst = np.sqrt((yPos - yMid)**2 + (xPos - xMid)**2)
                if(dst > maxDist):
                    maxDist = dst
            
        return maxDist
        
    
    def initializeImmuneSystem(self):
        self.immuneSystem = ImmuneAutomaton(self.width, self.height)
        self.immuneSystem.evasionProbability = 0.01
        
        #Set some parameters
        self.immuneSystem.bCellInflammation = 0.5
        self.immuneSystem.tCellInflammation = 1.5
        self.immuneSystem.helperCellInflammation = 0.5
        #self.immuneSystem.cytokineDissipation = 0.02*(self.ratioUpdateImmune/10)
        #self.immuneSystem.cytokineDiffusion = 0.1*(self.ratioUpdateImmune/10)
        
        #self.immuneSystem.rHelper = 1/self.ratioUpdateImmune
        #self.immuneSystem.rHelper = 0.05*(self.ratioUpdateImmune/10)
        #self.immuneSystem.rBCell = 0.1*(self.ratioUpdateImmune/10)
        self.immuneSystem.rAntibody = 1
        self.immuneSystem.rTAttack = 1
        #self.immuneSystem.evasionProbability = 0.02*(self.ratioUpdateImmune/10)
        self.immuneSystem.cytokineDissipation = 0.02
        self.immuneSystem.cytokineDiffusion = 0.1
        self.immuneSystem.rHelper = 0.1
        self.immuneSystem.rBCell = 0.1
        
        
        
        
        self.immuneSystem.updateCapCytokine()
        
    
    def countNeighbors(self, x, y):
        sumNeighbors = 0
        for i in range(-1,2):
            for j in range(-1,2):
                if(i != 0 or j != 0):
                    sumNeighbors = sumNeighbors + self.occupiedPositions[(y + i)%self.height, (x + j)%self.width]
        return sumNeighbors
    
    def getPositionsToInvade(self, x, y):
        positions = []
        for i in range(-1,2):
            for j in range(-1,2):
                index1 = (y + i)%self.height
                index2 = (x + j)%self.width
                if((i != 0 or j != 0) and self.occupiedPositions[index1,index2] == 0 and self.necroticPositions[index1,index2] == 0):
                    positions.append([index1,index2])
        
        return positions
    
    def updateNutrientAndECM(self):
        for i in range(0,self.height):
            for j in range(0,self.width):
                nNeighbors = self.countNeighbors(j,i)
                self.ecm.updateMatrix(nNeighbors, i, j)
                if(self.occupiedPositions[i,j] == 0):
                    self.nutrient.updateNutrient(None, j, i)
        
        
        #Update nutrient where cells are occupied
        for i in range(0,len(self.cells)):
            self.nutrient.updateNutrient(self.cells[i],self.cells[i].x, self.cells[i].y)
    
    def addProliferatingCell(self, x, y):
        newCell = Cell(x,y, CellType.PROLIFERATING)
        if(not newCell in self.cells):
            self.cells.append(newCell)
            self.occupiedPositions[y,x] = 1
    
    def makeTumorMalignant(self):
        self.inflammationResponseFactor = 1.5
        self.rProlif = 0.8
        self.setImmuneSuppresion(0.3)
    
    def makeTumorMalignantSet(self, inflammationResponseFactor):
        self.inflammationResponseFactor = inflammationResponseFactor
        self.rProlif = 0.8
        self.setImmuneSuppresion(0.3)
        
    
    def getImmunePicture(self):
        tCellColorInactive = [222/255, 29/255, 29/255]
        tCellColorActive = [204/255, 64/255, 255/255]
        tCells = self.immuneSystem.tCells
        picture = np.zeros((self.height, self.width, 3))
        for i in range(0,len(self.cells)):
            cell = self.cells[i]
            index1 = cell.y
            index2 = cell.x
            if(cell.cellType == CellType.PROLIFERATING or cell.cellType == CellType.COMPLEX):
                picture[index1, index2, :] = cell.cellType.value
        
        for i in range(0,len(tCells)):
            tCell = tCells[i]
            index1 = tCell.y
            index2 = tCell.x
            if(tCell.isActive()):
                picture[index1, index2, :] = tCellColorInactive
            
            else:
                picture[index1, index2, :] = tCellColorActive
                
        for i in range(0,np.size(self.necroticPositions,0)):
            for j in range(0,np.size(self.necroticPositions,0)):
                if(self.necroticPositions[i,j] == 1):
                    picture[i,j,:] = CellType.NECROTIC.value
        
        
        return picture
        
        
                
        
    
    def updateCells(self, step):
        cellsToDelete = []
        cellsToAdd = []
        
        #Cells are updated in random order
        indsList = list(range(0,len(self.cells)))
        random.shuffle(indsList)
        minCytokine = np.min(self.immuneSystem.cytokineConcentration)
        maxCytokine = np.max(self.immuneSystem.cytokineConcentration)
        for i in indsList:
            cell = self.cells[i]
            cell.countCycle = cell.countCycle + 1
            positionsToInvade = self.getPositionsToInvade(cell.x, cell.y)
            if(len(positionsToInvade) > 0 and cell.isQuiescent()):
                cell.setQuiescent(False)
            
            elif(len(positionsToInvade) == 0):
                cell.setQuiescent(True)
        
            oxygenConcentration = self.nutrient.getNutrientValue(cell.y, cell.x)
            cell.breathe(oxygenConcentration)
            
            #Cell updates from Immune system 
            if(self.immuneSystem.wasDisposed(cell.y, cell.x)):
                cell.cellType = CellType.DEAD
            
            elif(cell.cellType == CellType.PROLIFERATING and self.immuneSystem.isAntigenTrapped(cell.y, cell.x)):
                cell.cellType = CellType.COMPLEX
            
            elif(cell.cellType == CellType.COMPLEX and not self.immuneSystem.isAntigenTrapped(cell.y, cell.x)):
                cell.cellType = CellType.PROLIFERATING
                
            if(cell.cellType == CellType.PROLIFERATING and not cell.isQuiescent()):
                immuneFactor = self.getImmuneSystemFactor(cell.x, cell.y, minCytokine, maxCytokine)
                reproductionProb = np.min([self.rProlifPrime*immuneFactor, 1])
                
                if(random.random() < reproductionProb):
                    indsPositions = list(range(0,len(positionsToInvade)))
                    random.shuffle(indsPositions)
                    for p in indsPositions:
                        candidatePosition = positionsToInvade[p]
                        if(self.ecm.canInvadePosition(candidatePosition[0], candidatePosition[1]) and not candidatePosition in cellsToAdd):
                            cellsToAdd.append(candidatePosition)
                            break
                    
            elif(cell.cellType == CellType.DEAD):
                if(random.random() < self.rDecay):
                    cellsToDelete.append(cell)
            
            if(self.cells[i].cellType == CellType.NECROTIC):
                self.necroticPositions[cell.y, cell.x] = 1
                cellsToDelete.append(cell)
        
        for i in range(0,len(cellsToDelete)):
                self.removeCell(cellsToDelete[i])
        
        for i in range(0,len(cellsToAdd)):
            cellPosition = cellsToAdd[i]
            self.addProliferatingCell(cellPosition[1], cellPosition[0])
            
    def evolve(self, nSteps, tumorMovie, immuneMovie, immuneTumorMovie):
        self.cellCountSeries = np.zeros((nSteps+1, 4))
        self.tumorSizeSeries = np.zeros(nSteps+1)
        immuneMovieIndex = 0
        if(tumorMovie):
            self.tumorMovie = np.zeros((self.height, self.width, 3, nSteps + 1))
        if(immuneMovie):
            self.immuneMovie = np.zeros((self.height, self.width, 3, nSteps*self.ratioUpdateImmune + 1))
        if(immuneTumorMovie):
            self.immuneTumorMovie = np.zeros((self.height, self.width, 3, nSteps+1))
        
        for i in tqdm(range(0,nSteps)):
            counts = self.getCellCounts()
            tumorSize = self.getTumorRadius()
            #print(np.max(self.immuneSystem.cytokineConcentration))
            #print(np.max(self.immuneSystem.cytokineConcentration*self.immuneSystem.antibodyGrid))
            #print(self.immuneSystem.cytokineConcentration[20,20])
            self.cellCountSeries[i,:] = counts
            self.tumorSizeSeries[i] = tumorSize
            if(tumorMovie):
                self.tumorMovie[:,:,:,i] = self.getPicture(True)
            if(immuneTumorMovie):
                self.immuneMovie[:,:,:,i] = self.getImmunePicture()
            
            totalCells = counts[0]
            self.rProlifPrime = self.rProlif*(1 - totalCells/self.K)
            #print(totalCells)
            self.updateNutrientAndECM()
            self.updateTherapy(i)
            self.updateImmuneSystem(immuneMovie, immuneMovieIndex)
            immuneMovieIndex = immuneMovieIndex + self.ratioUpdateImmune
            self.updateCells(i)
            #print(counts[0]+counts[1])
            #print(len(self.immuneSystem.helperCells))
            #print(len(self.immuneSystem.bCells))
            #print(len(self.immuneSystem.tCells))
            #print("\n")
        
        self.cellCountSeries[nSteps, :] = self.getCellCounts()
        self.tumorSizeSeries[nSteps] = self.getTumorRadius()
        if(tumorMovie):
            self.tumorMovie[:,:,:,nSteps] = self.getPicture(True)
        if(immuneTumorMovie):
            self.immuneTumorMovie[:,:,:,nSteps] = self.getImmunePicture()
            
    
    def plotTumorSizeEvolution(self,ax):
        ax.plot(list(range(0,len(self.tumorSizeSeries))),self.tumorSizeSeries, color = "#1990e6")
        ax.set_xlabel("Step")
        ax.set_ylabel("Max. tumor radius")
    
    def getCellCounts(self):
        proliferatingCells = 0
        complexCells = 0
        necroticCells = 0
        deadCells = 0
        
        for i in range(0,len(self.cells)):
            cell = self.cells[i]
            if(cell.cellType == CellType.PROLIFERATING):
                proliferatingCells = proliferatingCells + 1
            elif(cell.cellType == CellType.COMPLEX):
                complexCells = complexCells + 1
            elif(cell.cellType == CellType.DEAD):
                deadCells = deadCells + 1
        
        necroticCells = sum(sum(self.necroticPositions))
        
        return np.array([proliferatingCells, complexCells, deadCells, necroticCells])
    
    
    
    def updateImmuneSystem(self, immuneMovie, index):
        
        antigenPositions = np.zeros((self.height, self.width))
        for i in range(0,len(self.cells)):
            cellType = self.cells[i].cellType
            if((cellType == CellType.PROLIFERATING and random.random() < 1 - self.rSuppresion) or cellType == CellType.COMPLEX):
                index1 = self.cells[i].y
                index2 = self.cells[i].x
                antigenPositions[index1, index2] = 1
        
        self.immuneSystem.resetAttackedPositions()
        
        
        self.immuneSystem.setAntigenPositions(antigenPositions)
        
        for i in range(0,self.ratioUpdateImmune):
            self.immuneSystem.stepImmuneAutomaton()
            if(immuneMovie):
                self.immuneMovie[:,:,:,index + i] = self.immuneSystem.getPicture() 
                
    
    def getImmuneSystemFactor(self, x, y,minCytokine, maxCytokine):
        index1 = y
        index2 = x
        #Inflammation will help the tumor grow
        cytokineConcentration = self.immuneSystem.cytokineConcentration[index1, index2]
        normalizedCytokine = (cytokineConcentration-minCytokine)/(maxCytokine - minCytokine)
        immuneSystemFactor = np.min([1 + normalizedCytokine*(self.maxProlifImmuneBoost - 1)*self.inflammationResponseFactor, self.maxProlifImmuneBoost])
        
        #if(immuneSystemFactor > 2):
           # print("Exceeded 2: "+str(immuneSystemFactor))
        
        return immuneSystemFactor
    
    def setImmuneSuppresion(self, rSuppresion):
        self.rSuppresion = rSuppresion
    
    
    def plotEvolution(self,ax):
        n = np.size(self.cellCountSeries, 0)
        ax.plot(self.cellCountSeries[:,0], color = CellType.PROLIFERATING.value, label = "Proliferating", linewidth = 2)
        ax.plot(self.cellCountSeries[:,1], color = CellType.COMPLEX.value, label = "Complex", linewidth = 2)
        ax.plot(self.cellCountSeries[:,2], color = CellType.DEAD.value, label = "Dead", linewidth = 2)
        ax.set_xlabel("Step")
        ax.set_ylabel("Number of cells")
        ax.legend()
        
    def getPicture(self, includeNecrotic):
        picture = np.zeros((self.height, self.width, 3))
        
        for i in range(0, len(self.cells)):
            picture[self.cells[i].y, self.cells[i].x,:] = self.cells[i].cellType.value
        if(includeNecrotic):
            for i in range(0, self.height):
                for j in range(0, self.width):
                    if(self.necroticPositions[i,j] == 1):
                        picture[i,j,:] = self.colorNecrotic
        
        return picture
            
    def removeCell(self, cell):
        self.cells.remove(cell)
        self.occupiedPositions[cell.y, cell.x] = 0
        
    #Therapy methods
    
    def addTherapy(self, therapy):
        self.therapies.append(therapy)
    
    def updateTherapy(self, step):
        if(len(self.therapies) > 0):
            for i in range(0,len(self.therapies)):
                self.therapies[i].updateTherapy(step, self)
    
    
        