import numpy as np
from dataprocessing import *
#box:[上, 左, 下, 右]

def DIoU(box1, box2, w_xy, angles_true, angles_pre):
    # 计算中间矩形的宽高
    Npos = 0
    L_reg_sum = 0
    Lcls_sum = 0
    for i in range(40):
        for j in range(40):
            if not np.all(box1[i,j,:]==0):
                Npos += 1
                box1[i,j,0] -= 4+i*8  #left
                box1[i,j,1] += 4+i*8  #top
                box1[i,j,2] += 4+i*8  #right
                box1[i,j,3] -= 4+i*8  #down
                box2[i,j,0] -= 4+i*8  #left
                box2[i,j,1] += 4+i*8  #top
                box2[i,j,2] += 4+i*8  #right
                box2[i,j,3] -= 4+i*8  #down
                in_h = min(box1[i,j,3], box2[i,j,3]) - max(box1[i,j,1], box2[i,j,1])
                in_w = min(box1[i,j,2], box2[i,j,2]) - max(box1[i,j,0], box2[i,j,0])
    
    # 计算交集、并集面积
                inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
                union = (box1[i,j,3] - box1[i,j,1]) * (box1[i,j,2] - box1[i,j,0]) + \
                (box2[i,j,3] - box2[i,j,1]) * (box2[i,j,2] - box2[i,j,0]) - inter
    # 计算IoU
                iou = inter / union
    # 计算对角线长度
                x1,y1,x2,y2 = box1[i,j,:4] 
                x3,y3,x4,y4 = box2[i,j,:4]
                C = np.sqrt((max(x1,x2,x3,x4)-min(x1,x2,x3,x4))**2 + \
                (max(y1,y2,y3,y4)-min(y1,y2,y3,y4))**2)
    # 计算中心点间距
                point_1 = ((x2+x1)/2, (y2+y1)/2)
                point_2 = ((x4+x3)/2, (y4+y3)/2)
                D = np.sqrt((point_2[0]-point_1[0])**2 + \
                (point_2[1]-point_1[1])**2)
    # 计算空白部分占比
                lens = D**2 / C**2
                diou = iou - lens

                L_reg = (1-diou)*w_xy[i,j]
                L_reg_sum += L_reg

                for k in range(19):
                    if angles_true[i,j,k] == 1:
                        Lcls = 0.25*((1-angles_pre[i,j,k])**2)*np.log(angles_pre[i,j,k])
                    else:  
                        Lcls = 0.75*((angles_pre[i,j,k])**2)*np.log(1-angles_pre[i,j,k])
                    Lcls_sum += Lcls
                Lcls_sum = -Lcls_sum
            Lcls_sum += Lcls_sum
    Lcls = Lcls_sum/Npos
    L_reg = 2/Npos*L_reg_sum
    L = Lcls + L_reg
    return L
box1 = target(points,angles_original)
box2 = target(points,angles_original)
L = DIoU(box1[0],box2[0],box1[1],box1[2],box2[2])
print(L)
