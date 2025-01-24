import open3d as o3d
import copy

def minVector (points_np):
     minVectorPoint= copy.deepcopy(points_np[0])
     for point in points_np:
       if point[0] < minVectorPoint[0]:
          minVectorPoint[0] = copy.copy(point[0])
       if point[1] < minVectorPoint[1]:
          minVectorPoint[1] = copy.copy(point[1])
       if point[2] < minVectorPoint[2]:
          minVectorPoint[2] = copy.copy(point[2])
     return minVectorPoint

def maxVector (points_np):
     maxVectorPoint= copy.deepcopy(points_np[0])
     for point in points_np:
       if point[0] > maxVectorPoint[0]:
          maxVectorPoint[0] = copy.copy(point[0])
       if point[1] > maxVectorPoint[1]:
          maxVectorPoint[1] = copy.copy(point[1])
       if point[2] > maxVectorPoint[2]:
          maxVectorPoint[2] = copy.copy(point[2])
     return maxVectorPoint

def cuboid3D_mmdetection3d_annotation_segmentsai (minPoint, maxPoint, category_name):
    centerBox=[(minPoint[0]+maxPoint[0])/2,(minPoint[1]+maxPoint[1])/2, (minPoint[2]+maxPoint[2])/2 ]
    dimensionBox= [abs(maxPoint[0]-minPoint[0]), abs(maxPoint[1]-minPoint[1]), abs(maxPoint[2]-minPoint[2])]
    headingAngle = 0
    return str(round(centerBox[0],2))+" "+str(round(centerBox[1],2))+" "+str(round(centerBox[2],2))+" "+str(round(dimensionBox[0],2))+" "+str(round(dimensionBox[1],2))+" "+str(round(dimensionBox[2],2))+" "+str(headingAngle)+" "+category_name
