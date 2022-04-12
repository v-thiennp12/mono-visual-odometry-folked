import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from monovideoodometery import MonoVideoOdometery
import os, uuid
from scipy.spatial.transform import Rotation

# kitti odom gray
# img_path    = 'C:\\Monocular-Video-Odometery\\data_odometry_gray\\dataset\\sequences\\00\\image_0\\'
# pose_path   = 'C:\\Monocular-Video-Odometery\\data_odometry_poses\\dataset\\poses\\00.txt'
# focal       = 718.8560
# pp          = (607.1928, 185.2157)
# frame_h         = 376
# frame_w         = 1241

#kitti 360
img_path    = 'C:\\fisheye_wrapping\\kitti360Scripts-master\\data\\sequence\\2013_05_28_drive_0000_sync_image_02\\image_0\\'
pose_path   = 'C:\\Monocular-Video-Odometery\\data_odometry_poses\\dataset\\poses\\2013_05_28_drive_0000_sync_image_02.txt'
focal       = 1336 # gamma1: 1.3363220825849971e+03
pp          = (716.94323510126321, 705.76498308221585 - 420)
frame_h     = 750
frame_w     = 1400

R_total     = np.zeros((3, 3))
t_total     = np.empty(shape=(3, 1))

# Parameters for lucas kanade optical flow
lk_params   = dict( winSize  = (21,21),
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

# Create some random colors
color   = np.random.randint(0,255,(10000,3))

vo      = MonoVideoOdometery(img_path, pose_path, focal, pp, lk_params)
trj_h   = 800
trj_w   = 800
traj    = np.zeros(shape=(trj_h, trj_w, 3))

# video saving
vid_name    = 'images\\demo' + uuid.uuid1().hex + '.avi'
# vid_h       = 376
# vid_w       = 1241# (376, 1241) (720, 1280)
# vid_size    = (vid_w, vid_h) # video width x video height
# frame_h     = vid_h
# frame_w     = vid_w
# vid_frame   = (np.ones((frame_h, frame_w, 3))*255).astype(np.uint8) # frame height x frame width

trj_h_rezized   = frame_h
trj_w_rezized   = int(trj_w*trj_h_rezized/trj_h)
vid_frame       = (np.ones((frame_h, frame_w + trj_w_rezized, 3))*255).astype(np.uint8) # frame height x frame width
vid_size        = (frame_w + trj_w_rezized, frame_h) # video width x video height
# cv.('M','J','P','G') 'mp4v' 'P','I','M','1' 'F','F','V','1' DIVX 'mjpg'
vid_writer      = cv.VideoWriter(vid_name, cv.VideoWriter_fourcc(*'DIVX'), 20, vid_size, True)

flag = True
while(vo.hasNextFrame()):
    
    frame       = vo.current_frame

    frame       = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    
    mask        = np.zeros((frame_h, frame_w, 3)).astype(np.uint8)
    
    for i, (new,old) in enumerate(zip(vo.good_new, vo.good_old)):
        a,b = new.ravel()    
        c,d = old.ravel()

        # print(a,b)
        # print(c,d)
        a, b = int(a), int(b)
        c, d = int(c), int(d)

        if np.linalg.norm(new - old) < 10:
            if flag:
                mask    = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                frame   = cv.circle(frame, (a,b), 2, color[i].tolist(), -1)
    
    cv.imshow('mask', mask)
    
    # cv.add(frame, mask)
    frame = cv.addWeighted(frame, 0.4, mask, 0.6, 0.0)
    cv.imshow('frame', frame)
    k = cv.waitKey(1)
    if k == 27:
        break

    if k == 121:
        flag = not flag
        toggle_out = lambda flag: "On" if flag else "Off"
        print("Flow lines turned ", toggle_out(flag))
        mask = np.zeros_like(vo.old_frame)
        mask = np.zeros_like(vo.current_frame)

    vo.process_frame()

    print(vo.get_mono_coordinates())

    mono_coord = vo.get_mono_coordinates()
    true_coord = vo.get_true_coordinates()

    print("MSE Error: ", np.linalg.norm(mono_coord - true_coord))
    print("x: {}, y: {}, z: {}".format(*[str(pt) for pt in mono_coord]))
    print("true_x: {}, true_y: {}, true_z: {}".format(*[str(pt) for pt in true_coord]))

    euler_angles_0 = (Rotation.from_matrix(vo.R)).as_euler('ZXY', degrees=True)
    print("roll : {}, pitch : {}, yaw : {}".format(*[str(pt) for pt in euler_angles_0]))
    print("t_x: {}, t_y: {}, t_z: {}".format(*[str(pt) for pt in mono_coord]))
    print('=====================================================================')

    draw_x, draw_y, draw_z = [int(round(x)) for x in mono_coord]
    true_x, true_y, true_z = [int(round(x)) for x in true_coord]

    traj = cv.circle(traj, (true_x + 400, true_z + 400), 1, list((0, 0, 255)), 4)
    traj = cv.circle(traj, (draw_x + 400, draw_z + 400), 1, list((0, 255, 0)), 4)

    cv.putText(traj, 'Actual Position:', (140, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Red', (270, 90), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 255), 1)
    cv.putText(traj, 'Estimated Odometry Position:', (30, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 1)
    cv.putText(traj, 'Green', (270, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 1)

    cv.imshow('trajectory', traj)

    #video save

    # frame_BRG = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
    frame_BRG   = frame

    # vid_frame = cv.resize(frame_BRG, (frame_w, frame_h))
    # trj_w4    = int(frame_w/4)
    # trj_h4    = min(int(traj.shape[0]*trj_w4/traj.shape[1]), int(frame_h))
    # vid_frame[:trj_h4, :trj_w4, :] = cv.resize(traj, (trj_w4, trj_h4))

    vid_frame[:frame_h, :frame_w, :] = frame_BRG
    vid_frame[:frame_h, frame_w:, :] = cv.resize(traj, (trj_w_rezized, trj_h_rezized))

    vid_writer.write(vid_frame.astype(np.uint8))

    # cv.imwrite("./images/trajectory__.png", vid_frame.astype(np.uint8))


vid_writer.release()

cv.imwrite("./images/trajectory" + uuid.uuid1().hex + ".png", traj)

cv.destroyAllWindows()