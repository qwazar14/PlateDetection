import ComputerVision, ParserCarsUa

def GetInfo():
    a = input() #
    value = ComputerVision.PlateReg(a)
    ParserCarsUa.CheckingOnWebsite(value)

GetInfo()
