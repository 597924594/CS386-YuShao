import numpy as np
import cv2
import scipy.ndimage
import growing
import pywt


def GaussianBlur_self_fit(pic_input):#pic is 48,48
    import math
    def get_pic_val(pic, x, y):
        x_lim, y_lim = pic.shape
        if x < 0 or x >= x_lim or y < 0 or y >= y_lim:
            return 0
        else:
            return pic[x, y]
    def U_x_y(pic, x, y, r, sigma_u):
        ans = 0
        for i in range(x-r, x+r):
            ans += math.exp(- ((get_pic_val(pic, i, y+r) - get_pic_val(pic, x, y)) ** 2) / (2 * sigma_u * sigma_u))
        for i in range(x-r, x+r):
            ans += math.exp(- ((get_pic_val(pic, i, y-r) - get_pic_val(pic, x, y)) ** 2) / (2 * sigma_u * sigma_u))
        for i in range(y-r+1, y+r-1):
            ans += math.exp(- ((get_pic_val(pic, x+r, i) - get_pic_val(pic, x, y)) ** 2) / (2 * sigma_u * sigma_u))
        for i in range(y-r+1, y+r-1):
            ans += math.exp(- ((get_pic_val(pic, x-r, i) - get_pic_val(pic, x, y)) ** 2) / (2 * sigma_u * sigma_u))
        ans = ans / (8*r)
        return round(ans, 4)
    def get_sigma_u(x, y, pic):
        grediant_list = []
        for i in range(x):
            for j in range(y):
                grediant_list.append(get_pic_val(pic, i+1, j) - get_pic_val(pic, i, j))
                grediant_list.append(get_pic_val(pic, i, j+1) - get_pic_val(pic, i, j))
        grediant_list = sorted(grediant_list)
        #grediant_list = grediant_list[0: int(0.9*len(grediant_list))]
        grediant_array = np.array(grediant_list)
        mean = grediant_array.sum() / len(grediant_list)
        var = (grediant_array*grediant_array).sum() / len(grediant_list) - mean * mean
        return round(mean + 3 * var, 4)
    pic = np.array(pic_input, dtype='float64')
    x, y = pic.shape
    sigma_u = get_sigma_u(x, y, pic)
    max_r = 10
    # r_list = [(max_r - i) for i in range(max_r - 1)]
    output = pic
    if sigma_u == 0: return output
    print 'sigma_u is %d' % sigma_u
    print '==============================================='
    for i in range(48):
        for j in range(48):
            R_xy = 1
            for r in range(1, max_r):
                Uxy = U_x_y(pic, i, j, r, sigma_u)
                if Uxy < 0.85:
                    R_xy = r
                    break
            # print ('R_xy for %d,%d is %d and Uxy is %d' % (i, j, R_xy, Uxy))
            #if R_xy == 0: continue
            pix_val = 0
            for x_i in range(i - R_xy, i + R_xy):
                for y_j in range(j - R_xy, j+R_xy):
                    pix_val += get_pic_val(pic, x_i, y_j) * math.exp(-((x_i-i)**2+(y_j-j)**2) / (2 * R_xy * R_xy)) / ((2 * math.pi)**0.5 * R_xy)
            output[i, j] = round(pix_val, 4)
    for i in range(48):
        for j in range(48):
            output[i, j] = round(output[i, j], 4)
    return output