from ast import If
import math
from pydoc import doc
from re import A
from sre_constants import IN
from unittest import result
import numdifftools as nd
import random
import numpy as np

def fun(Variable,R):
        result = 0 
        result =  -1*(((pow(math.sin(2*math.pi*Variable[0]), 3))*(math.sin(2*math.pi*Variable[1])))/((Variable[0]**3)*(Variable[0]+Variable[1])))
        return  (result + R*cons(Variable))

def cons(Variable):
        y1=  (-(Variable[0])**2 + (Variable[1]) - 1)
        y2=  ( -1 + (Variable[0]) - (Variable[1]-4)**2 )

        if y1 < 0:
            y1= y1 * y1
        else:
            y1=0

        if y2 < 0:
            y2 = y2 * y2
        else:
            y2=0

        return (y1+y2)



def grad(Variable,R):

    gradient_results = nd.Gradient(fun)(Variable,R)
    return gradient_results 

def exhaustive_search(input_variables,Variable, gradient,R):
    n = input_variables[1]
    a = -20
    b = 10
    h = (b-a)/n
    x1 = a
    x2 = x1 + h
    x3 = x2 + h
    Variable1 = Variable - x1*gradient
    Variable2 = Variable - x2*gradient
    Variable3 = Variable - x3*gradient
    f1=  fun(Variable1,R)
    f2 = fun(Variable2,R)
    f3 = fun(Variable3,R)
    #fuv_eval = fuv_eval +3 
    while(x3 <= b):
        #fuv_eval= fuv_eval + 1
        if(f1 >= f2 and f2 <= f3):            
            return x1, x3
        else:
            x1 = x2                                   #Updating points
            x2 = x3 
            x3 = x2 + h
            Variable3 = Variable - x3*gradient
            f1 = f2
            f2 = f3
            f3 = fun(Variable3,R)
    
    #print(fuv_eval,'\n')
    return [x1,x3]
   

def interval_halv(input_variables, Variable, gradient, lower, upper,R):
    error = 1/pow(10, input_variables[2])
    lo = abs(upper - lower)                  
    xm = (lower + upper)/2
    Variablen = Variable - xm*gradient
    fn = fun(Variablen,R)  
    while( lo >= error):
        #fuv_eval= fuv_eval+2
        A = lower + lo/4                  
        B = upper - lo/4
        Variable1 = Variable - A*gradient
        Variable2 = Variable - B*gradient
        f1 = fun(Variable1,R)
        f2 = fun(Variable2,R)
        #print(f1)
        if ( f1 < fn):
            upper = xm                     
            xm = A
            fn = f1
        elif( f2 < fn):
            lower = xm                      
            xm = B
            fn = f2
        else:
            lower = A                     
            upper = B
        
        lo = (B - A)

    return (A + B)/2

 
                  

file1 = open('Input_file.txt', 'r')
file2=open("output.txt",'w')
lines = file1.read().splitlines()
input = []                                         #create list file
for line in lines:
    if line.isdigit() == True:
        input.append(int(line))

input_variables = np.array(input)
num1 = input_variables[5] 
num2 = input_variables[6]
Variable =[num1,num2]
R=0.1  
print('The Initial values for the variables are :\n')
print('The Initial values for the variables are :',file=file2)
print(Variable, '\n')
print(Variable, '',file=file2)
print('The Initial Value of Objective Function is:')
print(np.absolute(fun(Variable,R)))
#print(cons(Variable))
m = input_variables[3]
n = input_variables[4]
eps1 = 1/(10**n)
eps2 = pow(10, -10)
gradient = grad(Variable,R)
#print(gradient)
k = 0  
fuv_eval = 0  
print('iteration \t fuction value',file=file2)

while(np.linalg.norm(gradient) >= eps1):
    print(k,'\t\t',np.absolute(fun(Variable,R)),file=file2)
    
    if k < m:
        Variable_pre = Variable
        lower,upper = exhaustive_search(input_variables, Variable, gradient,R )
        alpha = interval_halv(input_variables, Variable, gradient, lower, upper,R)
        Variable = Variable - (alpha * gradient)
        R= (10 * R)
        norm_value = np.linalg.norm(Variable - Variable_pre)/np.linalg.norm(Variable_pre)
        if(np.linalg.norm(np.dot(grad(Variable,R), grad(Variable_pre,(R/10))))) <= eps2:
            print('***************************')
            break
        if norm_value <= eps2:
            break
        else:
            k = k+1                   
            gradient = grad(Variable,R)
    else:
        break


print('The no. of iteration  are :\n')
print('The no. of iteration  are :',file=file2)   
print(k,file=file2)
print(k)
print("**************************************************************")
print("**************************************************************",file=file2)
print('The Finals values for the variables are :\n')
print('The Finals values for the variables are :',file=file2)
final_result  = np.round(Variable, 5)
print(np.absolute(final_result))
print(np.absolute(final_result),file=file2)
print('The Final Value of Objective Function is:')
print('The Final Value of Objective Function is:',file=file2)
print(np.absolute(round(fun(Variable,R),4)))
print(np.absolute(round(fun(Variable,R),4)),file=file2)
file2.close()


