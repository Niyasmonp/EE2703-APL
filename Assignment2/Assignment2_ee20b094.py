"""
 Applied Programming Lab 2703 - 2022
 Assignment 2 
 Name    : Niyas Mon P
 Roll No : EE20B094
"""

from cmath import exp
from sys import argv , exit
from numpy import *

FILE_TYPE = ".netlist"  #file type
START     = ".circuit"  #start of spice code
END       = ".end"      #end of the spice code
AC = ".ac"

class Resistor():
    def __init__(self,fromNode,toNode,value) -> None:
        self.name = "resistor"
        self.fromNode = fromNode
        self.toNode = toNode
        self.value = float(value)

class Inductor():
    def __init__(self,fromNode,toNode,value) -> None:
        self.name = "inductor"
        self.fromNode = fromNode
        self.toNode = toNode
        self.value = float(value)

class Capacitor():
    def __init__(self,fromNode,toNode,value) -> None:
        self.name = "capacitor"
        self.fromNode = fromNode
        self.toNode = toNode
        self.value = float(value)

class dcVoltageSource():
    def __init__(self,fromNode,toNode,value) -> None:
        self.name = "dc voltage source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.amp = float(value)
        self.phase = 0

class dcCurrentSource():
    def __init__(self,fromNode,toNode,value) -> None:
        self.name = "dc current source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.amp = float(value)
        self.phase = 0

class acVoltageSource():
    def __init__(self,fromNode,toNode,amp,phase) -> None:
        self.name = "dc voltage source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.amp = float(amp)/2
        self.phase = float(phase)

class acCurrentSource():
    def __init__(self,fromNode,toNode,amp,phase) -> None:
        self.name = "dc current source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.amp = float(amp)/2
        self.phase = float(phase)

class VCVS():
    def __init__(self,fromNode,toNode,controllingVoltageFromNode,controllingVoltageToNode,value) -> None:
        self.name = "voltage controlled voltage source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingVoltageFromNode = controllingVoltageFromNode
        self.controllingVoltageToNode = controllingVoltageToNode
        self.value = float(value)

class VCCS():
    def __init__(self,fromNode,toNode,controllingVoltageFromNode,controllingVoltageToNode,value) -> None:
        self.name = "voltage controlled current source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingVoltageFromNode = controllingVoltageFromNode
        self.controllingVoltageToNode = controllingVoltageToNode
        self.value = float(value)

class CCVS():
    def __init__(self,fromNode,toNode,controllingVoltageSource,value) -> None:
        self.name = "current controlled voltage source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingVoltageSource = controllingVoltageSource
        self.value = float(value)

class CCCS():
    def __init__(self,fromNode,toNode,controllingVoltageSource,value) -> None:
        self.name = "current controlled current source"
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingVoltageSource = controllingVoltageSource
        self.value = float(value)

# function to solve Ax=b
def solve_linalg(A,b):
    try:
        solution = linalg.solve(A,b)
    except:
        solution = False
        print("Error , check the input matrices again")
    return solution

# function for extracting lines in ckt definition and frequency
def readFile(file_name):
    try:

        f = open(file_name)  
        lines = f.readlines()  #reading the file 
        f.close()              #closing the file

        Afreq=0 #variable to store angular frequency

        #getting the index of line of '.circuit' and '.end'
        for line in lines:
            if AC == line[:len(AC)]:
                Afreq = 2*pi*float(line.split()[2])    
            if START == line[:len(START)]:
                line_start = lines.index(line)

            elif END == line[:len(END)]:
                line_end = lines.index(line)
                # break


        if line_start >= line_end:   #checking the correct circuit definition
            print("Invalid circuit definition")
            exit(0)
        

        #list to store extracted lines
        lines_extracted=[]  
        for line in lines[line_start+1:line_end]:
            words = line.strip().split("#")[0].split()      #removing comments and splitting the line into words
            line = " ".join(words)      #joining the words into lines back after removing the comments
            if len(line) > 0:
                lines_extracted.append(line)    #appending the modified line to a new list      

        return lines_extracted , Afreq


    except:
        print("Invalid file")
        exit()


#function to extract tokens and store the objects of classes of each component in seperate lists
#this function also stores nodes in a dictionary called "Nodes"
def analyseTokens(lines):
    Nodes = {"GND":0}
    R = [] #resistor
    L = [] #inductor
    C = [] #capacitor
    V = [] #voltage source
    I = [] #current source
    E = [] #VCVS
    G = [] #VCCS
    H = [] #CCVS
    F = [] #CCCS
    i=1

    for line in lines:        
        tokens = line.split()
        # extracting the resistor
        if tokens[0][0] == 'R':

            R.append(Resistor(tokens[1],tokens[2],tokens[3]))

            if not tokens[1] in Nodes.keys(): #storing the nodes in dictionary
                Nodes[tokens[1]]= i  
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1
        
        # extracting the capacitor
        if tokens[0][0] == 'C':
            C.append(Capacitor(tokens[1],tokens[2],tokens[3]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1

        # extracting the inductor
        if tokens[0][0] == 'L':
            L.append(Inductor(tokens[1],tokens[2],tokens[3]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1

        # extracting the voltage source
        if tokens[0][0] == 'V':
            if tokens[3] == "dc" :
                V.append(dcVoltageSource(tokens[1],tokens[2],tokens[4]))

                if not tokens[1] in Nodes.keys():
                    Nodes[tokens[1]]= i
                    i = i+1
                if not tokens[2] in Nodes.keys():
                    Nodes[tokens[2]]= i
                    i = i+1
            if tokens[3] == "ac" :
                V.append(acVoltageSource(tokens[1],tokens[2],tokens[4],tokens[5]))

                if not tokens[1] in Nodes.keys():
                    Nodes[tokens[1]]= i
                    i = i+1
                if not tokens[2] in Nodes.keys():
                    Nodes[tokens[2]]= i
                    i = i+1

        # extracting the current source
        if tokens[0][0] == 'I':
            if tokens[3] == "dc":
                I.append(dcCurrentSource(tokens[1],tokens[2],tokens[4]))

                if not tokens[1] in Nodes.keys():
                    Nodes[tokens[1]]= i
                    i = i+1
                if not tokens[2] in Nodes.keys():
                    Nodes[tokens[2]]= i
                    i = i+1
            if tokens[3] == "ac":
                I.append(dcCurrentSource(tokens[1],tokens[2],tokens[4],tokens[5]))

                if not tokens[1] in Nodes.keys():
                    Nodes[tokens[1]]= i
                    i = i+1
                if not tokens[2] in Nodes.keys():
                    Nodes[tokens[2]]= i
                    i = i+1

        # extracting the ccvs
        if tokens[0][0] == 'H':
            H.append(CCVS(tokens[1],tokens[2],tokens[3],tokens[4]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1

        # extracting the cccs
        if tokens[0][0] == 'F':
            F.append(CCCS(tokens[1],tokens[2],tokens[3],tokens[4]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1

        # extracting the vcvs
        if tokens[0][0] == 'E':
            E.append(VCVS(tokens[1],tokens[2],tokens[3],tokens[4],tokens[5]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1
            if not tokens[3] in Nodes.keys():
                Nodes[tokens[3]]= i
                i = i+1
            if not tokens[4] in Nodes.keys():
                Nodes[tokens[4]]= i
                i = i+1

        # extracting the vccs
        if tokens[0][0] == 'G':
            G.append(VCCS(tokens[1],tokens[2],tokens[3],tokens[4],tokens[5]))

            if not tokens[1] in Nodes.keys():
                Nodes[tokens[1]]= i
                i = i+1
            if not tokens[2] in Nodes.keys():
                Nodes[tokens[2]]= i
                i = i+1
            if not tokens[3] in Nodes.keys():
                Nodes[tokens[3]]= i
                i = i+1
            if not tokens[4] in Nodes.keys():
                Nodes[tokens[4]]= i
                i = i+1

    return Nodes , R , L , C , V , I , E , G , H , F


# function for defining matrix of appropriate size
def Matrix(n_nodes,V, E, H):
    m = n = n_nodes + len(V) + len (E) + len(H)
    M = zeros((m,n),dtype=complex)
    b = zeros(m,dtype=complex)
    return M ,b


# function for adding elements in matrix_b
def add_b(II,VV,b_matrix):
    for ii in II:
        b_matrix[Nodes[ii.fromNode]] += -ii.amp *exp(1j*ii.phase)
        b_matrix[Nodes[ii.toNode]] += ii.amp *exp(1j*ii.phase)
    for vv in VV:
        b_matrix[n_nodes + VV.index(vv)] += vv.amp *exp(1j*vv.phase)
        

    return b_matrix


# function for adding elements in matrix_G
def add_element(Res,ind,cap,vol,vccs,vcvs,cccs,ccvs,matrix):

    for RR in Res:     #adding resistors
        matrix[Nodes[RR.fromNode]][Nodes[RR.fromNode]] += 1/RR.value 
        matrix[Nodes[RR.fromNode]][Nodes[RR.toNode]] += -1/RR.value
        matrix[Nodes[RR.toNode]][Nodes[RR.fromNode]] += -1/RR.value
        matrix[Nodes[RR.toNode]][Nodes[RR.toNode]] += 1/RR.value

    for LL in ind:    #adding inductors
        matrix[Nodes[LL.fromNode]][Nodes[LL.fromNode]] += 1/(1j*LL.value*angfreq) 
        matrix[Nodes[LL.fromNode]][Nodes[LL.toNode]] += -1/(1j*LL.value*angfreq)
        matrix[Nodes[LL.toNode]][Nodes[LL.fromNode]] += -1/(1j*LL.value*angfreq)
        matrix[Nodes[LL.toNode]][Nodes[LL.toNode]] += 1/(1j*LL.value*angfreq)

    for CC in cap:   #adding capacitors
        matrix[Nodes[CC.fromNode]][Nodes[CC.fromNode]] += (1j*CC.value*angfreq) 
        matrix[Nodes[CC.fromNode]][Nodes[CC.toNode]] += (-1j*CC.value*angfreq)
        matrix[Nodes[CC.toNode]][Nodes[CC.fromNode]] += (-1j*CC.value*angfreq)
        matrix[Nodes[CC.toNode]][Nodes[CC.toNode]] += (1j*CC.value*angfreq)

    for VV in vol:  #adding voltage sources
        matrix[Nodes[VV.fromNode]][n_nodes+vol.index(VV)] += -1 
        matrix[Nodes[VV.toNode]][n_nodes+vol.index(VV)] += 1
        matrix[n_nodes+vol.index(VV)][Nodes[VV.fromNode]] += 1
        matrix[n_nodes+vol.index(VV)][Nodes[VV.toNode]] += -1
        
    for GG in vccs:  #adding vccs
        matrix[Nodes[GG.fromNode]][Nodes[GG.controllingVoltageFromNode]] += GG.value
        matrix[Nodes[GG.fromNode]][Nodes[GG.controllingVoltageToNode]] += -GG.value
        matrix[Nodes[GG.toNode]][Nodes[GG.controllingVoltageFromNode]] += -GG.value
        matrix[Nodes[GG.toNode]][Nodes[GG.controllingVoltageToNode]] += GG.value

    for EE in vcvs:  #adding vcvs
        matrix[Nodes[EE.fromNode]][n_nodes+len(V) +vcvs.index(EE)] += 1
        matrix[Nodes[EE.toNode]][n_nodes+ len(V)+vcvs.index(EE)] += -1
        matrix[n_nodes+len(V)+ vcvs.index(EE)][EE.fromNode] += 1
        matrix[n_nodes+ len(V)+vcvs.index(EE)][EE.toNode] += -1
        matrix[n_nodes+ len(V)+vcvs.index(EE)][EE.controllingVoltageFromNode] += -EE.value
        matrix[n_nodes+len(V)+ vcvs.index(EE)][EE.controllingVoltageToNode] += EE.value

    for FF in cccs:   #adding cccs
        x= n_nodes + int(FF.controllingVoltageSource[1]) -1 
        matrix[Nodes[FF.fromNode]][x] += FF.value
        matrix[Nodes[FF.toNode]][x] += -FF.value

    for HH in ccvs:  #adding  ccvs
        y= n_nodes + int((HH.controllingVoltageSource[1])) -1
        matrix[Nodes[HH.fromNode]][n_nodes + len(V) + len(E) + ccvs.index(HH)] +=1
        matrix[Nodes[HH.toNode]][n_nodes + len(V) + len(E) + ccvs.index(HH)] += -1
        matrix[n_nodes + len(V) + len(E) + ccvs.index(HH)][Nodes[HH.fromNode]] += 1
        matrix[n_nodes + len(V) + len(E) + ccvs.index(HH)][Nodes[HH.toNode]] += -1
        matrix[n_nodes + len(V) + len(E) + ccvs.index(HH)][y] += -HH.value

    matrix[0] = 0
    matrix[0][0] = 1

    return matrix


# function to print solutions
def printSolution(x):
    print(f"Voltage at node GND  : 0 ")
    for n in range(1,len(x)):
        if(n < n_nodes):
            print(f"Voltage at node {Nodes[str(n)]}  : ", end=' ')
            print(x[n])
        else:
            print(f"Current through voltage source{n-n_nodes+1} : ", end=' ')
            print(x[n])




#check if the correct number of arguments given or not
if len(argv) !=2:  
    print("Invalid number of arguments given!") 
    exit()

#check the correct file type
file_name = argv[1]  
if file_name[-len(FILE_TYPE):] != FILE_TYPE:
    print("\nIncorrect file type. Only '.netlist' files are accepted.")
    exit()


if __name__=="__main__":

    lines , angfreq= readFile(file_name)
    Nodes, R, L, C , V, I, E, G, H, F = analyseTokens(lines)  #extracting tokens and storing them aprropriatley
    n_nodes= len(Nodes)

    M1,B1 = Matrix(n_nodes, V, E,H)    #creating matrix_G and matrix_b
    matrix_G = add_element(R,L,C,V,G,E,F,H,M1)  #adding elements to matrix_G
    matrix_b = add_b(I,V,B1)  ##adding elements to matrix_b
    matrix_x=(solve_linalg(matrix_G,matrix_b))  #solving the circuit
    printSolution(matrix_x)   #printing the solved voltages and currents


    