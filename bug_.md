_, R, t, _ = cv2.recoverPose(E, self.good_old, self.good_new, self.R.copy(), self.t.copy(), focal=self.focal, pp=self.pp, mask=None)

# 
zfill(6) >> kitti gray odom dataset
zfill(10) >> kitti 360 dataset