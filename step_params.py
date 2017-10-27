# ==============================================================================
# A step-by-step example using numpy of the example in tech details on how
# X3DNA computes base-pair step parameters.
# author: mauricio esguerra
# update: 26, October, 2017
# ==============================================================================
import numpy as np
from numpy import linalg

# ==============================================================================
# Compute the reference frame, origin o, and orientation R, for G1 and its
# paired base C8 in 2d94
# ==============================================================================
# Standard (S) base for G
S = np.loadtxt("S_G.dat")
# Experimental (E) base, from pdb file, G1.
E = np.loadtxt("E_G1.dat")
N = S.shape[0]
i = np.ones((N,1))
C = 1.0/(N-1.0) * (np.dot(S.T,E)-((1.0/N)*np.dot(np.dot(S.T,i), np.dot(i.T,E))))

M = np.array(
    [[C[0,0]+C[1,1]+C[2,2],   C[1,2]-C[2,1],          C[2,0]-C[0,2],          C[0,1]-C[1,0]],
     [C[1,2]-C[2,1],          C[0,0]-C[1,1]-C[2,2],   C[0,1]+C[1,0],          C[2,0]+C[0,2]],
     [C[2,0]-C[0,2],          C[0,1]+C[1,0],         -C[0,0]+C[1,1]-C[2,2],   C[1,2]+C[2,1]],
     [C[0,1]-C[1,0],          C[2,0]+C[0,2],          C[1,2]+C[2,1],         -C[0,0]-C[1,1]+C[2,2] ]]
)

w,l = linalg.eigh(M)
q = -1*(l.T[3])

R_G1 = np.array(
    [
    [q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*(q[1]*q[2]-q[0]*q[3]),                 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[2]*q[1]+q[0]*q[3]),                 q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[3]*q[1]-q[0]*q[2]),                 2*(q[3]*q[2]+q[0]*q[1]),                 q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]
    ]
)    

o_G1 = np.mean(E, axis=0) - np.dot(np.mean(S, axis=0),R_G1.T)

#print o_G1
#print R_G1


# ==================================================================
# Calculation for C8 in 2d94
# ==================================================================
# Standard (S) base for C
S = np.loadtxt("S_C.dat")
# Experimental (E) base, from pdb file, C8.
E = np.loadtxt("E_C8.dat")
N = S.shape[0]
i = np.ones((N,1))
C = 1.0/(N-1.0) * (np.dot(S.T,E)-((1.0/N)*np.dot(np.dot(S.T,i), np.dot(i.T,E))))

M = np.array(
    [[C[0,0]+C[1,1]+C[2,2],   C[1,2]-C[2,1],          C[2,0]-C[0,2],          C[0,1]-C[1,0]],
     [C[1,2]-C[2,1],          C[0,0]-C[1,1]-C[2,2],   C[0,1]+C[1,0],          C[2,0]+C[0,2]],
     [C[2,0]-C[0,2],          C[0,1]+C[1,0],         -C[0,0]+C[1,1]-C[2,2],   C[1,2]+C[2,1]],
     [C[0,1]-C[1,0],          C[2,0]+C[0,2],          C[1,2]+C[2,1],         -C[0,0]-C[1,1]+C[2,2] ]]
)

w,l = linalg.eigh(M)
q = -1*(l.T[3])

R_C8 = np.array(
    [
    [q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*(q[1]*q[2]-q[0]*q[3]),                 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[2]*q[1]+q[0]*q[3]),                 q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[3]*q[1]-q[0]*q[2]),                 2*(q[3]*q[2]+q[0]*q[1]),                 q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]
    ]
)    

o_C8 = np.mean(E, axis=0) - np.dot(np.mean(S, axis=0),R_C8.T)

#print o_C8
#print R_C8


# ==================================================================
# Calculation for G2 in 2d94
# ==================================================================
# Standard (S) base for C
S = np.loadtxt("S_G.dat")
# Experimental (E) base, from pdb file, C8.
E = np.loadtxt("E_G2.dat")
N = S.shape[0]
i = np.ones((N,1))
C = 1.0/(N-1.0) * (np.dot(S.T,E)-((1.0/N)*np.dot(np.dot(S.T,i), np.dot(i.T,E))))

M = np.array(
    [[C[0,0]+C[1,1]+C[2,2],   C[1,2]-C[2,1],          C[2,0]-C[0,2],          C[0,1]-C[1,0]],
     [C[1,2]-C[2,1],          C[0,0]-C[1,1]-C[2,2],   C[0,1]+C[1,0],          C[2,0]+C[0,2]],
     [C[2,0]-C[0,2],          C[0,1]+C[1,0],         -C[0,0]+C[1,1]-C[2,2],   C[1,2]+C[2,1]],
     [C[0,1]-C[1,0],          C[2,0]+C[0,2],          C[1,2]+C[2,1],         -C[0,0]-C[1,1]+C[2,2] ]]
)

w,l = linalg.eigh(M)
q = -1*(l.T[3])

R_G2 = np.array(
    [
    [q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*(q[1]*q[2]-q[0]*q[3]),                 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[2]*q[1]+q[0]*q[3]),                 q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[3]*q[1]-q[0]*q[2]),                 2*(q[3]*q[2]+q[0]*q[1]),                 q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]
    ]
)    

o_G2 = np.mean(E, axis=0) - np.dot(np.mean(S, axis=0),R_G2.T)

#print o_G2
#print R_G2


# ==================================================================
# Calculation for C7 in 2d94
# ==================================================================
# Standard (S) base for C
S = np.loadtxt("S_C.dat")
# Experimental (E) base, from pdb file, C7.
E = np.loadtxt("E_C7.dat")
N = S.shape[0]
i = np.ones((N,1))
C = 1.0/(N-1.0) * (np.dot(S.T,E)-((1.0/N)*np.dot(np.dot(S.T,i), np.dot(i.T,E))))

M = np.array(
    [[C[0,0]+C[1,1]+C[2,2],   C[1,2]-C[2,1],          C[2,0]-C[0,2],          C[0,1]-C[1,0]],
     [C[1,2]-C[2,1],          C[0,0]-C[1,1]-C[2,2],   C[0,1]+C[1,0],          C[2,0]+C[0,2]],
     [C[2,0]-C[0,2],          C[0,1]+C[1,0],         -C[0,0]+C[1,1]-C[2,2],   C[1,2]+C[2,1]],
     [C[0,1]-C[1,0],          C[2,0]+C[0,2],          C[1,2]+C[2,1],         -C[0,0]-C[1,1]+C[2,2] ]]
)

w,l = linalg.eigh(M)
q = -1*(l.T[3])

R_C7 = np.array(
    [
    [q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*(q[1]*q[2]-q[0]*q[3]),                 2*(q[1]*q[3]+q[0]*q[2])],
    [2*(q[2]*q[1]+q[0]*q[3]),                 q[0]*q[0]-q[1]*q[1]+q[2]*q[2]-q[3]*q[3], 2*(q[2]*q[3]-q[0]*q[1])],
    [2*(q[3]*q[1]-q[0]*q[2]),                 2*(q[3]*q[2]+q[0]*q[1]),                 q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3]]
    ]
)    

o_C7 = np.mean(E, axis=0) - np.dot(np.mean(S, axis=0),R_C7.T)

#print C
#print type(C)
#print M
#print type(M)
#print q
#print o_C7
#print R_C7


# =================================================================
# Standard Reference frames for base-pairs
# =================================================================

o_G1C8 = (o_G1 + o_C8)/2
R_G1C8 = (R_G1 + R_C8)/2

o_G2C7 = (o_G2 + o_C7)/2
R_G2C7 = (R_G2 + R_C7)/2

print "o_G1C8 = ", o_G1C8
print "R_G1C8 = ", R_G1C8
print "o_G2C7 = ", o_G2C7
print "R_G2C7 = ", R_G2C7

# Hinge axis 
#h = cross(R_G2C7(:,1) - R_G1C8(:,1), R_G2C7(:,2) - R_G1C8(:,2)) 
#     n=( 1 / sqrt(dot(h,h)));
#     h_norm = n*h'

# RollTilt angle
#     dot(h_norm, h_norm);



