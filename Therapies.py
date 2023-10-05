
from enum import Enum

class TherapyType(Enum):
    
    IMMUNE_SUPPRESSOR_THERAPY = 0

    
class Therapy:
    
    def __init__(self,therapyType, *args):
        
        self.startStep = args[0]
        self.therapyType = therapyType
        
        if(therapyType == TherapyType.IMMUNE_SUPPRESSOR_THERAPY):
            self.tCellReductionEfficacy = args[1]
            self.attackReductionEfficacy = args[2]
            self.eliminationProbabilityPerCell = args[3]
            
    def updateTherapy(self, step, tumor):
        
        if(self.therapyType == TherapyType.IMMUNE_SUPPRESSOR_THERAPY):
            if(step == self.startStep):
                tumor.immuneSystem.rTAttack = tumor.immuneSystem.rTAttack*(1 - self.attackReductionEfficacy)
                tumor.immuneSystem.maxNTCells = tumor.immuneSystem.maxNTCells*(1 - self.tCellReductionEfficacy)
                tumor.immuneSystem.eliminationProb = self.eliminationProbabilityPerCell
        
        
            
    
    
    