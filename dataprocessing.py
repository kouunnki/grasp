import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.feature import peak_local_max


path = '/home/hyunqi/anaconda3/ggcnn-master/dataset/01/pcd0116cpos.txt'
def load_from_cornell_file_angles(fname, pad_to=0):
    """
    :return: Nx4 numpy array [left top x,left top y,right down x,right down y], N is the number of rectangles
    """
    grs = []
    with open(fname) as f:
        while True:
            # Load 4 lines at a time, corners of bounding box.
            p0 = f.readline()
            if not p0:
                break  # EOF
            p1, p2, p3 = f.readline(), f.readline(), f.readline()
            try:
               x0, y0 = [int(float(s)) for s in p0.split()]  #left top
               x1, y1 = [int(float(s)) for s in p1.split()]  #right top
               x2, y2 = [int(float(s)) for s in p2.split()]  #right down
               x3, y3 = [int(float(s)) for s in p3.split()]  #left down
               gr = np.array([x0, y0, x1, y1,x2,y2,x3,y3])
               grs.append(gr)

            except ValueError:
                # Some files contain weird values.
                continue
                
    a = np.stack(grs)
    return a.astype(np.int)
points_angles = load_from_cornell_file_angles(fname=path)


def load_from_cornell_file(fname, pad_to=0):
    """
    :return: Nx4 numpy array [left top x,left top y,right down x,right down y], N is the number of rectangles
    """
    grs = []
    with open(fname) as f:
        while True:
            # Load 4 lines at a time, corners of bounding box.
            p0 = f.readline()
            if not p0:
                break  # EOF
            p1, p2, p3 = f.readline(), f.readline(), f.readline()
            try:
               x0, y0 = [int(float(s)) for s in p0.split()]  #left top
               x1, y1 = [int(float(s)) for s in p1.split()]  #right top
               x2, y2 = [int(float(s)) for s in p2.split()]  #right down
               x3, y3 = [int(float(s)) for s in p3.split()]  #left down
               gr = np.array([x0, y0,x2, y2,])
               grs.append(gr)

            except ValueError:
                # Some files contain weird values.
                continue
                
    a = np.stack(grs)
    return a.astype(np.int)
points = load_from_cornell_file(fname=path)


def caculate_center(points):
        """
        Compute mean center of all GraspRectangles
        :return: float, mean centre of all GraspRectangles
        """
        return np.mean(np.vstack(points), axis=0).astype(np.int)

def center_allrectangles(points):
        points = points.reshape((-1,2))
        return np.mean(points,axis=0).astype(np.int)


def _get_crop_attrs(points,output_size):
        center_all = caculate_center(points_angles)
        left = max(0, min(center_all[1] - output_size // 2, 640 - output_size))
        top = max(0, min(center_all[0] - output_size // 2, 480 - output_size))
        return center_all, left, top
center_all, left, top = _get_crop_attrs(points,320)

def angle(points):
        """
        :return: Angle of the grasp to the horizontal.Nx5
        """
        dx = points[:,0] - points[:,2]
        dy = points[:,1] - points[:,3]
        angles = ((np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2).reshape((-1,1))
        return angles
angles_original = angle(points_angles)
angles = angle(points_angles)


def rotate(point,angle):
        """
        Rotate grasp rectangle
        :param angle: Angle to rotate (in radians)
        :param center: Point to rotate around (e.g. image center)
        """

        R = np.array(
            [
                [np.cos(angle), np.sin(angle)],
                [-1 * np.sin(angle), np.cos(angle)],
            ]
        )
        
        R = np.squeeze(R)
        point = point.reshape((2,2))
        center = point.mean(axis=0).astype(np.int)
        c = np.array(center).reshape((1, 2))
        points = ((np.dot(R, (point - c).T)).T + c).astype(np.int)
        
        return points.ravel()

def CHV(points,angles):
    for i in range(len(angles)):
        if angles[i] >= 0 and angles[i] <= np.pi/4:
            angles[i] = -angles[i]
            points[i,:] = rotate(points[i,:],angles[i])
        elif angles[i] > np.pi/4 and angles[i] <= np.pi/2:
            angles[i] = np.pi/2 - angles[i]
            points[i,:] = rotate(points[i,:],angles[i])
        elif angles[i] < 0 and angles[i] >= -np.pi/4:
            angles[i] = -angles[i]
            points[i,:] = rotate(points[i,:],angles[i])
        elif angles[i] < -np.pi/4 and angles[i] >= -np.pi/2:
            angles[i] = -np.pi/2 - angles[i]
            points[i,:] = rotate(points[i,:],angles[i])
    return points
points = CHV(points,angles)

def zoom(points, factor, center):
        """
        Zoom grasp rectangle by given factor.
        :param factor: Zoom factor
        :param center: Zoom zenter (focus point, e.g. image center)
        """
        T = np.array(
            [
                [1/factor, 0],
                [0, 1/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        points = points.reshape((-1,2))
        points = ((np.dot(T, (points - c).T)).T + c).astype(np.int)
        return points.reshape((-1,4))
points = points.reshape((-1,2))
points += np.array((-top,-left)).reshape((1, 2))
points = points.reshape((-1,4))
middle_center = center_allrectangles(points)
points = zoom(points,factor=1.0,center=middle_center)

def target(gt_boxes,angles):
    gt_boxes = np.array(gt_boxes, np.int32)
    raw_height, raw_width = 320,320
    gt_boxes_area = (np.abs(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])))  #the area of the boxes
    boxes_cnt = len(gt_boxes)                                                    #the number of the boxes

    shift_x = np.arange(0, raw_width).reshape(-1, 1)
    shift_y = np.arange(0, raw_height).reshape(-1, 1)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  #create a XY-coordinate system

    off_l = (shift_x[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])  #print(off_l.shape)  (320, 320, 6, 1)
    off_t = (shift_y[:, :, np.newaxis, np.newaxis] -
             gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
    off_r = -(shift_x[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
    off_b = -(shift_y[:, :, np.newaxis, np.newaxis] -
              gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])
    
    offset = np.concatenate([off_l, off_t, off_r, off_b], axis=3)

    fm_height = 40
    fm_width = 40
    stride = 8
    shift_x = np.arange(0, fm_width)
    shift_y = np.arange(0, fm_height)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    xy = np.vstack((shift_y.ravel(), shift_x.ravel())).transpose()

    off_xy = offset[xy[:, 0] * stride, xy[:, 1] * stride]  #print:(1600, 6, 4)
    raw_xy = xy*stride+stride/2        #(1600,2) the coordinate of 1600 points in the input picture
    raw_xy_1 = raw_xy.reshape((40,40,2))
    w = np.abs(gt_boxes[:, 2] - gt_boxes[:, 0])   #(1,6)
    h = np.abs(gt_boxes[:, 3] - gt_boxes[:, 1])
    w = np.tile(w, (1600, 1))                #(1600,6)
    h = np.tile(h, (1600, 1))  
    center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    center_y = (gt_boxes[:, 1] + gt_boxes[:, 3])/ 2.0  #(1,6)
    center_x = np.tile(center_x, (1600, 1))
    center_y = np.tile(center_y, (1600, 1))            #(1600,6)
    delta_x = np.abs(raw_xy[:, 1][:, np.newaxis] - center_x)    #(1600,6)
    delta_y = np.abs(raw_xy[:, 0][:, np.newaxis] - center_y)

    is_in_center = np.all([delta_x,delta_y]<=np.array([w/4,h/4]), axis=0)  #(1600,6)
    rectangles = is_in_center.astype(int)
    
    dw2 = (rectangles*delta_x)**2
    dh2 = (rectangles*delta_y)**2
    distance = dw2 + dh2

    #set all non-zero numbers to zero except minimum in every row
    for i in range(distance.shape[0]):
        row = distance[i,:]
        nonzero_idx = np.nonzero(row)[0]
        if len(nonzero_idx) > 0:
            min_idx = np.argmin(row[nonzero_idx])
            row[nonzero_idx[min_idx]] = np.min(row[nonzero_idx])
            row[nonzero_idx[:min_idx]] = 0
            row[nonzero_idx[min_idx+1:]] = 0
    
    distance[np.where(distance != 0)] = 1    #distance here is the position of the selected points 标注了所有点选择的矩形
    dw2_weight = dw2*distance
    dh2_weight = dh2*distance
    mask = distance != 0  # 创建一个布尔型掩码，只计算非零数
    rx2 = ((w * distance) / 6) ** 2
    ry2 = ((h * distance) / 6) ** 2
    pre_w_xy = np.zeros_like(distance)  # 创建一个与 distance 大小相同的全0数组
    pre_w_xy[mask] = -(dw2_weight[mask] / (2 * rx2[mask]) + dh2_weight[mask] / (2 * ry2[mask]))
    w_xy = np.where(pre_w_xy != 0,np.exp(pre_w_xy),0).reshape((40,40,-1))
    w_xy = np.sum(w_xy,axis=2)

    off_xy_pre = np.expand_dims(distance, axis=2)
    off_xy_pre = np.repeat(off_xy_pre, 4, axis=2)
    off_xy = off_xy_pre*off_xy
    off_xy = np.sum(off_xy,axis=1).reshape((40,40,4))
    
    angles_class = np.round(((angles+np.pi/(180/85))/(np.pi/18)+1),decimals=0)
    angles_class = np.tile(angles_class.T,(1600,1))
    angles_class = np.sum(angles_class*distance,axis=1).reshape((40,40))
    #off_xy = np.concatenate([off_xy, angles_class[..., np.newaxis]], axis=-1)
    angles_one_hot = np.eye(19)[angles_class.astype(int) ]
    angles_one_hot[angles_class == 0] = 0
    return off_xy,w_xy,angles_one_hot
x = target(points,angles_original)
np.set_printoptions(threshold=np.inf)
#print(np.sum(x))
#plt.imshow(x, cmap='hot',interpolation='nearest')
#plt.savefig('./myplot.png')
#我想判断每个锚点是否在矩形的中心区域，具体实现的方法是判断4[dw,dh]≤[w,h]，其中dw,dh是锚点与矩形中心的距离，w,h是矩形的宽和高
#这里还是有些问题，得到的is_in_center全是0，因为图片大小不正确，在论文中提到的图片大小都是320x320，在代码中也是如此计算，但是实际的图片并不是这么大，导致像是x，y的中心坐标都是错误的，因此需要将图片resize之后再重新读入数据