
from enum import Enum
import numpy as np
import random
from TumorModel import CellType

class TherapyType(Enum):
    
    IMMUNE_SUPPRESSOR_THERAPY = 0
    RADIOTHERAPY = 1
    CHEMOTHERAPY = 2
    

    
class Therapy:
    
    def __init__(self,therapyType, *args):
        
        self.startStep = args[0]
        self.therapyType = therapyType
        self.inheritanceResistanceProbability = 0.6
        self.necrosisTherapyRate = 0.15
        
        if(therapyType == TherapyType.IMMUNE_SUPPRESSOR_THERAPY):
            self.tCellReductionEfficacy = args[1]
            self.attackReductionEfficacy = args[2]
            self.eliminationProbabilityPerCell = args[3]
        
        elif(therapyType == TherapyType.RADIOTHERAPY):
            self.g0Gamma = args[1]
            self.alpha = args[2]
            self.beta = args[3]
            self.cycleTime = args[4]
            
            self.dose = args[5]
            self.thresholdOxygen = args[6]
            self.delayTime = args[7]
            self.initMitoticProb = args[8]
            self.finalMitoticProb = args[9]
            self.therapyPeriod = args[10]
        
        elif(therapyType == TherapyType.CHEMOTHERAPY):
            self.treatmentResistances = args[1]
            self.killRates = args[2]
            self.attenuationCoefficients = args[3]
            self.nTreatmentSteps = args[4]
            self.tau = args[5]
            self.PK = args[6]
            self.widthAreaTreatment = args[7]
            self.heightAreaTreatment = args[8]
            self.initialMedConcentration = args[9]
            self.resistanceCellsRatio = args[10]
            self.medAbsorptionCells = args[11]
            self.medDiffusionConstant = args[12]
            self.medConcentration = np.zeros((self.heightAreaTreatment, self.widthAreaTreatment))
            self.medConcentration[0,:] = self.initialMedConcentration
            self.medConcentration[:,0] = self.initialMedConcentration
            self.medConcentration[:,self.widthAreaTreatment - 1] = self.initialMedConcentration
            self.medConcentration[self.heightAreaTreatment-1,:] = self.initialMedConcentration
            self.applicationSteps = args[13]
            
    def updateTherapy(self, step, tumor):
        
        if(self.therapyType == TherapyType.CHEMOTHERAPY):
            if(step - self.startStep >= 0):
                
                if(step - self.startStep == self.applicationSteps):
                    self.medConcentration[0,:] = 0
                    self.medConcentration[:,0] = 0
                    self.medConcentration[self.heightAreaTreatment-1,:] = 0
                    self.medConcentration[:,self.widthAreaTreatment - 1] = 0
                        
                    
                for i in range(1,self.heightAreaTreatment-1):
                    for j in range(1,self.widthAreaTreatment-1):
                        laplacian = self.medConcentration[i+1,j] + self.medConcentration[i-1,j] + self.medConcentration[i,j+1] + self.medConcentration[i,j-1] - 4*self.medConcentration[i,j]
                        deltaConcentration = self.medDiffusionConstant*laplacian - self.medAbsorptionCells
                        self.medConcentration[i,j] = self.medConcentration[i,j] + deltaConcentration
                    

                cells = tumor.cells
                for i in range(0,len(cells)):
                    cell = cells[i]
                    if(step == self.startStep or cell.countCycle == 0):
                        if(random.random() < 1 - self.resistanceCellsRatio):
                            cell.chemoAffected = True
                        else:
                            cell.chemoAffected = False
            
                    if(cell.chemoAffected and not cell.cellType == CellType.PROLIFERATING):
                        cellType = cell.cellType
                        killRatio = self.killRates[cellType]
                        resistance = self.treatmentResistances[cellType]
                        attenuationCoefficient = self.attenuationCoefficients[cellType]
                        concentration = self.medConcentration[cell.y,cell.x]
                        li = (killRatio*concentration)/(resistance*self.nTreatmentSteps + 1)
                        probKill = li*self.PK*np.exp(-attenuationCoefficient*(step - self.startStep-self.nTreatmentSteps*self.tau))
                        cellCycle = cell.countCycle%4
                
                        if(random.random() < probKill and cellCycle == 1):
                            if(random.random() < self.necrosisTherapyRate):
                                cell.turnNecrotic()
                            else:
                                cell.cellType = CellType.DEAD
        
        
        elif(self.therapyType == TherapyType.IMMUNE_SUPPRESSOR_THERAPY):
            if(step == self.startStep):
                tumor.immuneSystem.rTAttack = tumor.immuneSystem.rTAttack*(1 - self.attackReductionEfficacy)
                tumor.immuneSystem.maxNTCells = int(tumor.immuneSystem.maxNTCells*(1 - self.tCellReductionEfficacy))
                tumor.immuneSystem.eliminationProb = self.eliminationProbabilityPerCell
                
        
        elif(self.therapyType == TherapyType.RADIOTHERAPY):
            if((step - self.startStep)>= 0 and (step - self.startStep) % self.therapyPeriod == 0):
                cells = tumor.cells
                countAffect = 0
                for i in range(0,len(cells)):
                    cell = cells[i]
                    stageCellCycle = (cell.countCycle % self.cycleTime)//(self.cycleTime//4)
                    gamma = self.g0Gamma*(1.5)**(stageCellCycle)
                    index1 = cell.y
                    index2 = cell.x
                    oxygenConcentration = tumor.nutrient.nutrientConcentration[index1, index2]
                    oer = 0
                    if(oxygenConcentration > self.thresholdOxygen):
                        oer = 1
                    else:
                        oer = 1-(oxygenConcentration/self.thresholdOxygen)
                
                    dOER = self.dose/oer
                
                    probTarget = 1 - np.exp(-gamma*(self.alpha*dOER + self.beta*dOER**2))
                    if(random.random() < probTarget):
                        countAffect = countAffect + 1
                        cell.radioAffected = True
            
            elif(step - self.startStep >= 0):
                cells = tumor.cells
                for i in range(0,len(cells)):
                    cell = cells[i]
                    if(cell.radioAffected and cell.cellType == CellType.PROLIFERATING):
                        if((step - self.startStep)%self.therapyPeriod < self.delayTime):
                            if(random.random() < self.initMitoticProb):
                                    
                                if(random.random() < self.necrosisTherapyRate):
                                    cell.turnNecrotic()
                                else:
                                    cell.cellType = CellType.DEAD
                                    
            
                        elif((step-self.startStep)%self.therapyPeriod >= self.delayTime):
                            if(random.random() < self.finalMitoticProb):
                                if(random.random() < self.necrosisTherapyRate):
                                    cell.turnNecrotic()
                                else:
                                    cell.cellType = CellType.DEAD
                                    
                
                
                    
                
        
            
    
    
    