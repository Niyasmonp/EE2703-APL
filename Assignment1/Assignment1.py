'''
        EE2703 Applied Programming Lab-2022
        Assignment 1: Solution
        Name : Niays Mon P
        Roll No :EE20B094
'''

from sys import argv, exit


FILE_TYPE = ".netlist"  #file type
START     = ".circuit"  #start of spice code
END       = ".end"      #end of the spice code



#check if the correct number of arguments given or not
if len(argv) !=2:  
    print("Invalid number of arguments given!") 
    exit()


#check the correct file type
file_name = argv[1]  
if file_name[-len(FILE_TYPE):] != FILE_TYPE:
    print("\nIncorrect file type. Only '.netlist' files are accepted.")
    exit()


try:

    f = open(file_name)  
    lines = f.readlines()  #reading the file 
    f.close()              #closing the file


    #getting the index of line of '.circuit' and '.end'
    for line in lines:     
        if START == line[:len(START)]:
            line_start = lines.index(line)

        elif END == line[:len(END)]:
            line_end = lines.index(line)

            if line_start >= line_end:   #checking the correct circuit definition
                print("Invalid circuit definition")
                exit(0)
            break


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

            # if we want to print the analysed tokens and the names/nodes/values uncomment the following line
            # print(analyseTokens(line))

    #reversing the lines 
    lines_new = lines_extracted[:]#[line_start+1:line_end]
    lines_new.reverse()


    #reversing the words and finally printing desired output
    for line in lines_new:  
        words = line.split()
        words.reverse()
        line = " ".join(words)
        print(line)


except:
    print("Invalid file")
    exit()


# function to analyze the tokens and determine the from node, the to node, the type of element and the value.
def analyseTokens(line):
    tokens=line.split()

    # R, L, C, Independent Sources
    if(len(tokens) == 4):
        elementName = tokens[0]
        node1 = tokens[1]
        node2 = tokens[2]
        value = tokens[3]
        return [elementName, node1, node2, value]
    
    # CCCS/CCVS
    elif(len(tokens) == 5):
        elementName = tokens[0]
        node1 = tokens[1]
        node2 = tokens[2]
        voltageSource = tokens[3]
        value = tokens[4]
        return [elementName, node1, node2, voltageSource, value]

    # VCVS/VCCS
    elif(len(tokens) == 6):
        elementName = tokens[0]
        node1 = tokens[1]
        node2 = tokens[2]
        voltageSourceNode1 = tokens[3]
        voltageSourceNode2 = tokens[4]
        value = tokens[5]
        return [elementName, node1, node2, voltageSourceNode1, voltageSourceNode2, value]

    else:
        return []

"""
Brief overview:
lines are splitted and words separated and reversed and printed
So, in brief :
Checked proper netlist file is there or not
Removed comments
Splitted the words
Reversed lines
Reversed words
And printed everything in reverse order

In C code, We can do similar blocks of code but it will take more lines of code.
Codes like sort function may be inefficient compared to the similar functions in python.
C code is also not capable of handling incorrect inputs becuase not a interpreter file.
This can be handled and is an advantage of in Python.
"""
