import numpy as np 
import math

#sigmoid Function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Loss Function
def sse(A,B):
  dif = 1/2* (A-B)**2
  return dif

# Input data 
inputs = [0.05,0.10]

#real values
t1=0.01
t2=0.99

#initial weights this will happen randomly in real world
weights1 = [0.15]
weights2 = [0.20]
weights3 = [0.25]
weights4 = [0.30]
weights5 = [0.40]
weights6 = [0.45]
weights7 = [0.50]
weights8 = [0.55]

#initial weights this will initiate with zero
bias1 = 0.35
bias2 = 0.60

#neuron h1 and h2
h1 = inputs[0]*weights1[0]+inputs[1]*weights2[0]+bias1 
h2 = inputs[0]*weights3[0]+inputs[1]*weights4[0]+bias1

#activation function
out_h1=sigmoid(h1)
out_h2=sigmoid(h2)

#output
y1=out_h1*weights5[0]+out_h2*weights6[0]+bias2
y2=out_h1*weights7[0]+out_h2*weights8[0]+bias2

#activation function
out_y1=sigmoid(y1)
out_y2=sigmoid(y2)


#loss function 
e_total= sse(t1,out_y1)+sse(t2,out_y2)


#Applying Gradent On weights third layer
de_total_by_Dw5=-(t1-out_y1)*out_y1*(1-out_y1)*out_h1
de_total_by_Dw6=-(t1-out_y1)*out_y1*(1-out_y1)*out_h2
de_total_by_Dw7=-(t2-out_y2)*out_y2*(1-out_y2)*out_h1
de_total_by_Dw8=-(t2-out_y2)*out_y2*(1-out_y2)*out_h2

#Gradent Optimiser third layer
weights5_updated=weights5[0]-0.5*de_total_by_Dw5
weights6_updated=weights6[0]-0.5*de_total_by_Dw6
weights7_updated=weights7[0]-0.5*de_total_by_Dw7
weights8_updated=weights8[0]-0.5*de_total_by_Dw8

#Applying Gradent On weights second layer
d_e1_by_out_h1=-((t1-out_y1)*out_y1*(1-out_y1))*weights5[0]
d_e2_by_out_h1=-(t2-out_y2)*out_y2*(1-out_y2)*weights7[0]
d_e1_by_out_h2=-((t1-out_y1)*out_y1*(1-out_y1))*weights6[0]
d_e2_by_out_h2=-(t2-out_y2)*out_y2*(1-out_y2)*weights8[0]

de_total_by_out_h1= d_e1_by_out_h1 + d_e2_by_out_h1
de_total_by_out_h2= d_e1_by_out_h2 + d_e2_by_out_h2

out_h1_by_h1 = out_h1*(1-out_h1)
out_h2_by_h2 = out_h2*(1-out_h2)

d_total_by_w1= de_total_by_out_h1* out_h1_by_h1 * inputs[0]
d_total_by_w2= de_total_by_out_h1* out_h1_by_h1 * inputs[1]
d_total_by_w3= de_total_by_out_h2* out_h2_by_h2 * inputs[0]
d_total_by_w4= de_total_by_out_h2* out_h2_by_h2 * inputs[1]

#Gradent Optimiser third layer
weights1_updated=weights1[0]-0.5*d_total_by_w1
weights2_updated=weights2[0]-0.5*d_total_by_w2
weights3_updated=weights3[0]-0.5*d_total_by_w3
weights4_updated=weights4[0]-0.5*d_total_by_w4
print(weights4_updated)