# Author: Akash Deo
# Computer Vision CS 6384.002 Project 1 Program 2  
# what program does: 
# Write a program that gets as input a color image, performs histogram equalization in the Luv domain, and
# writes the scaled image asi output. Histogram equalization in Luv is applied to the luminance values, as
# computed in the specified window. It requires a discretization step, where the real-valued L is discretized
# into 101 values.
# As in the second program, all L values below the smallest L value in the window should be mapped to
# 0, and all L value above the largest L value in the window should be mapped to 100.

# Divide by zero has been handled by using appropriate if conditions and also range is handled by converting rgb greater than
# 255 to 255 and rbg values less than 0 are set to 0.

import cv2
import numpy as np
import sys
import math

def invGamma(x):
    if x < 0.03928:
        x1 = x/12.92
    else:
        x1 = ((x + 0.055)/1.055)**(2.4)
    return x1

def gammaCorrection(x):
    if x < 0.00304:
        x1 = 12.92*x*255.0
    else:
        x1 = (1.055*(x**(1.0/2.4)) - 0.055)*255.0
    return x1

def rgbtoluv(b,g,r):
    b1 = b/255.0
    g1 = g/255.0
    r1 = r/255.0
    b1x = invGamma(b1)
    g1x = invGamma(g1)
    r1x = invGamma(r1)
    X = 0.412453*r1x + 0.357580*g1x + 0.180423*b1x
    Y = 0.212671*r1x + 0.715160*g1x + 0.072169*b1x
    Z = 0.019334*r1x + 0.119193*g1x + 0.950227*b1x
    u = 0
    v = 0
    u1 = 0
    v1 = 0
    xw = 0.95
    yw = 1
    zw = 1.09
    uw = 4*xw / (xw + 15*yw + 3*zw)
    vw = 9*yw / (xw + 15*yw + 3*zw)
    t = Y / yw
    L = 0
    if t > 0.008856:
        L = (116*(t)**(1.0/3.0)) - 16
    else:
        L = 903.3 * t
    d = X + 15*Y + 3*Z
    if d!=0:
        u1 = 4*X / d
        v1 = 9*Y / d
        u = 13 * L * (u1 - uw)
        v = 13 * L * (v1 - vw)

    return [L, u, v]


def luvtorgb(L, u, v):
    xw = 0.95
    yw = 1
    zw = 1.09
    uw = 4*xw / (xw + 15*yw + 3*zw)
    vw = 9*yw / (xw + 15*yw + 3*zw)
    u1 = 0
    v1 = 0
    if L == 0:
        u1 = 0
        v1 = 0
    else:
        u1 = (u + 13*uw*L) / (13*L)
        v1 = (v + 13*vw*L) / (13*L)
    Y = 0
    if L > 7.9996:
        Y = (((L + 16)/116.0)**3)*yw
    else:
        Y = (L*yw / 903.3)
    X = 0
    Z = 0
    if(v1 == 0):
        X = 0
        Z = 0
    else:
        X = (2.25 * Y * u1) / v1
        Z = Y*(3.0 - 0.75*u1 - 5*v1) / v1
    Rsrgb = 3.240479 * X - 1.53715 * Y - 0.498535 * Z
    Gsrgb = -0.969256 * X + 1.875991 * Y + 0.041556 * Z
    Bsrgb = 0.055648 * X - 0.204043 * Y + 1.057311 * Z
    R = int(gammaCorrection(Rsrgb))
    G = int(gammaCorrection(Gsrgb))
    B = int(gammaCorrection(Bsrgb))
    if (R > 255):
        R = 255
    if (G > 255):
        G = 255
    if (B > 255):
        B = 255
    if (R < 0):
        R = 0
    if (B < 0):
        B = 0
    if (G < 0):
        G = 0
    return [B, G, R]

#main function    

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]


if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

cv2.imshow("input image: " + name_input, inputImage)

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

luv = inputImage.astype(np.float64)
tmp1 = inputImage.astype(np.float64)
out1 = inputImage.astype(np.float64)
out2 = np.copy(inputImage)

for i in range(0, rows) :
    for j in range(0, cols) :
        b, g, r = inputImage[i, j]
        luv[i, j] = rgbtoluv(b, g, r)

Lmax = float("-inf")
Lmin = float("inf")
for i in range(H1, H2) :
    for j in range(W1, W2) :
        L, u, v = luv[i, j]
        if L < Lmin:
            Lmin = L
        if L > Lmax:
            Lmax = L

for i in range(0, rows):
    for j in range(0, cols):
        L, u, v = luv[i, j]
        L1 = L
        if L < Lmin:
            L1 = 0.0
        if L > Lmax:
            L1 = 100.0
        tmp1[i, j] = L1, u, v

buckets = []
f = []
f1 = []
floor1 = []
for i in range(0, 101):
    buckets.append(0)
    f.append(0)
    f1.append(0)
    floor1.append(0)

for i in range(H1, H2):
    for j in range(W1, W2):
        L, u, v = tmp1[i, j]
        L1 = int(round(L))
        buckets[L1] += 1

for i in range(len(buckets)):
    if i == 0:
        f[i] = (buckets[i])
    else:
        f[i] = (f[i-1] + buckets[i])

for i in range(len(buckets)):
    if i == 0:
        f1[i] = ((f[i]/2)*(len(buckets)/f[len(f)-1]))
    else:
        f1[i] = (((f[i-1] + f[i])*len(buckets))/(2 * f[len(f)-1]))

for i in range(len(buckets)):
    floor1[i] = (math.floor(f1[i]))
    if floor1[i]>100:
        floor1[i] = 100
    if floor1[i]<0:
        floor1[i] = 0

for i in range(0, rows):
    for j in range(0, cols):
        L, u, v = tmp1[i, j]
        L2 = 0
        L1 = int(round(L))
        L2 = floor1[L1]
        out1[i, j] = L2, u, v


for i in range(0, rows):
    for j in range(0, cols):
        L, u, v = out1[i, j]
        out2[i, j] = luvtorgb(L, u, v)

cv2.imshow('final2', out2)
cv2.imwrite(name_output, out2);
# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()