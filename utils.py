import pickle
import numpy as np
import matplotlib.pylab as plt


class Transform:
    def __init__(self, r=np.eye(3, dtype='float'), t=np.zeros(3, 'float'), s=np.ones(3, 'float')):
        self.r = r.copy()
        self.t = t.reshape(-1).copy()
        self.s = s.copy()

    def __mul__(self, other):
        r = np.dot(self.r, other.r)
        t = np.dot(self.r, other.t * self.s) + self.t
        if not hasattr(other, 's'):
            other.s = np.ones(3, 'float').copy()
        s = other.s.copy()
        return Transform(r, t, s)

    def inv(self):
        r = self.r.T
        t = - np.dot(self.r.T, self.t)
        return Transform(r, t)

    def transform(self, xyz):
        if not hasattr(self, 's'):
            self.s = np.ones(3, 'float').copy()
        assert xyz.shape[-1] == 3
        assert len(self.s) == 3
        return np.dot(xyz * self.s, self.r.T) + self.t

    def getmat4(self):
        M = np.eye(4)
        M[:3, :3] = self.r * self.s
        M[:3, 3] = self.t
        return M

class MultiCamParams:
    def __init__(self, root_dir, cam_list):
        self.root_dir = root_dir
        self.cam_list = cam_list
        self.action = None
        self.intr = None
        self.kinect_extr = None
        self.extr = None
        self.trans_dict = None
        self.subject_calib_dict = {
            'subject02': '1024',
        }

    def set_action(self, action):
        self.action = action
        date = self.subject_calib_dict[self.action.split('_')[0]]
        calib_dir = '%s/calib_%s' % (self.root_dir, date)
        self.intr = pickle.load(open('%s/intrinsic_param.pkl' % calib_dir, 'rb'))
        # print(self.intr.keys())
        self.kinect_extr = pickle.load(open('%s/kinect_extrinsic_param.pkl' % calib_dir, 'rb'))
        # print(self.kinect_extr.keys())
        self.extr = pickle.load(open('%s/extrinsic_param_%s.pkl' % (calib_dir, date), 'rb'))
        # print(self.extr.keys())
        print('[%s] %s' % (action, calib_dir))
        self.trans_dict = self.extrinsic_transform()

    def extrinsic_transform(self):
        trans_dict = {}
        # depth to color within each kinect
        for cam in self.cam_list:
            if 'kinect' in cam:
                trans_dict['%s_cd' % cam] = \
                    util.Transform(r=self.kinect_extr['%s_d2c' % cam][0], t=self.kinect_extr['%s_d2c' % cam][1] / 1000)

        # extrinsic from other cam depth to azure_kinect_0 depth
        key = 'azure_kinect_0-azure_kinect_0'
        trans_dict[key] = util.Transform(r=np.eye(3), t=np.zeros([3]))
        key = 'azure_kinect_0-azure_kinect_1'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        key = 'azure_kinect_0-azure_kinect_2'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)

        key = 'azure_kinect_0-kinect_v2_1'
        T_tmp = util.Transform(r=self.extr['azure_kinect_1-kinect_v2_1'][0],
                               t=self.extr['azure_kinect_1-kinect_v2_1'][1] / 1000)
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_1'] * T_tmp
        key = 'azure_kinect_0-kinect_v2_2'
        T_tmp = util.Transform(r=self.extr['azure_kinect_2-kinect_v2_2'][0],
                               t=self.extr['azure_kinect_2-kinect_v2_2'][1] / 1000)
        trans_dict[key] = trans_dict['azure_kinect_0-azure_kinect_2'] * T_tmp

        key = 'event_camera-azure_kinect_0'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        key = 'polar-azure_kinect_0'
        trans_dict[key] = util.Transform(r=self.extr[key][0], t=self.extr[key][1] / 1000)
        return trans_dict

    def get_extrinsic_transform(self, source, target, depth_cam=True):
        if '%s-%s' % (target, source) in self.trans_dict.keys():
            T_depth = self.trans_dict['%s-%s' % (target, source)]
            if depth_cam or 'kinect' not in target:
                return T_depth
            else:
                T_color = self.trans_dict['%s_cd' % target] * T_depth
                return T_color
        elif '%s-%s' % (source, target) in self.trans_dict.keys():
            T_depth = self.trans_dict['%s-%s' % (source, target)].inv()
            if depth_cam or 'kinect' not in target:
                return T_depth
            else:
                T_color = self.trans_dict['%s_cd' % target] * T_depth
                return T_color
        else:
            raise ValueError('[error] %s-%s not exist.' % (target, source))

    def get_intrinsic_param(self, cam):
        if cam in self.intr.keys():
            return self.intr[cam]
        else:
            raise ValueError('[error] %s not exist.' % cam)


# others
def projection(xyz, intr_param, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
