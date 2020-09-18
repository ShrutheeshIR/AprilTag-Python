import cv2
import numpy as np
import yaml
import sys
sys.path.insert(1, 'lib')
from apriltag import apriltag
import time
# import inspect

# argspec = inspect.getargspec(apriltag)
# print(dir(apriltag))
detector = apriltag("tag36h11", 1,1, 0.5, -0.5, 1,0)
TCO = np.load('/home/olorin/Desktop/IISc/aruco/aruco-markers/pose_estimation/TCO.npy')
startflag = 0
zero_pose = np.eye(4)
# imagepath = 'test.jpg'
cap = cv2.VideoCapture(4)

with open('/home/olorin/Desktop/IISc/hand-pose/catkin_ws/src/hand_pose/src/camera_calib.yaml') as f:
    camdict = yaml.safe_load(f)

# Camera Params
cammat = np.array(camdict['camera_matrix']['data']).reshape(3,3)
dist_coeffs = np.array(camdict['distortion_coefficients']['data'])
aprilctr = 0
aprilpos = np.zeros((100000,4,4))
apriltime = np.zeros((100000,1))

prevtrans = np.zeros((3,1))
while True:
    ret, frame = cap.read()
    tx = time.time()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    outimg = frame.copy()
# image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    detections = detector.detect_and_getpose(image, 0.038, 6.9328859319012781e+02, 6.9569408192473725e+02, 3.2837683124795666e+02, 2.4273285538697331e+02)
    corners = []
    if len(detections):
        print(len(detections), detections[0]['error'])
        # print(detections[0]['error'])
        rot = detections[0]['rot']
        # rot = rot @ np.array([
        #                 [1, 0, 0],
        #                 [0, 0, 1],
        #                 [0,-1, 0],
        #             ])
        # print("ROT", rot[2][2])
        trans = detections[0]['trans']
        # print(rot, trans)
        # print(np.linalg.norm(trans*100 - prevtrans*100))
        prevtrans = trans.copy()
        corners.append(np.array(detections[0]['corners'][None, ...], dtype = np.float32))
        # kjas
    #     print(dir(detector))
        # print(detections[0].keys())
    #     print(type(detections[0]))
    #     print(detections)
    # print(type(corners[0][0][0][0]))
    # print(corners[0].shape)
        
        ids = np.asarray([[1]], dtype = np.int32)
        
        # print(ids.shape)or 0 < rot[1][1] < 1
        outimg = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if detections[0]['error'] > 100 :
        # if 0 < rot[1][1] < 1:

            print("CONT")
            # continue
            # pass
        # if :
        #     pass
        else:
            transmatrix = np.eye(4)
            transmatrix[:3,:3] = rot
            transmatrix[:3,3] = np.squeeze(trans)
            transmatrix = np.matmul(np.linalg.inv(TCO), transmatrix)
            transmatrix[:3, 3] = transmatrix[:3,3] - zero_pose[:3, 3]
            # transmatrix = np.matmul(np.linalg.inv(zero_pose),transmatrix)
            if startflag:
                aprilpos[aprilctr] = transmatrix.copy()
                apriltime[aprilctr] = tx
                aprilctr += 1
            cv2.aruco.drawAxis(outimg, cammat, dist_coeffs, rot, trans, 0.1)
            cv2.putText(outimg, str(trans), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, 2)
    cv2.imshow('outimg', outimg)
    ch = cv2.waitKey(1)

    if ch==ord('q'):
        break

    if ch == ord('s'):

        startflag=1
        zero_pose = transmatrix.copy()


print(aprilpos[:aprilctr])
print(apriltime[:aprilctr] - apriltime[0])
# np.save('aprilpos.npy', aprilpos[:aprilctr])
# np.save('apriltime.npy', apriltime[:aprilctr])
# print('%.15f' %(apriltime[0]))
    # else:
    #     print("ERROR")