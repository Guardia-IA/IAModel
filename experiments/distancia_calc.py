import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

def hamming_distance(id1, id2, aruco_dict):
    b1 = aruco_dict.bytesList[id1].ravel()
    b2 = aruco_dict.bytesList[id2].ravel()
    xor = np.bitwise_xor(b1, b2)
    return sum(bin(x).count('1') for x in xor)

max_dist = 0
best_id = 0

for j in range(1, 250):
    dist = hamming_distance(0, j, dictionary)
    if dist > max_dist:
        max_dist = dist
        best_id = j

print(f"Respecto al ID 0, el más diferente es ID {best_id}")
print(f"Distancia Hamming: {max_dist}/36 bits")