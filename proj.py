import numpy as np
import scipy as sp
from scipy import ndimage
from itertools import izip
import math


def index_coords(data, origin=None):
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def reproject_image_into_polar(data, origin=None):
        ny, nx = data.shape[:2]
        if origin is None:
                origin = (nx//2, ny//2)
        x, y = index_coords(data, origin=origin)
        r, theta = cart2polar(x, y)

        # Make a regular (in polar space) grid based on the min and max r & theta
        nr = int(r.max()-r.min())
        nt = ny
        r_i = np.linspace(r.min(), r.max(), nr)
        theta_i = np.linspace(theta.min(), theta.max(), nt)
        theta_grid, r_grid = np.meshgrid(theta_i, r_i)
        
        xi, yi = polar2cart(r_grid, theta_grid)
        xi += origin[0]
        yi += origin[1]

        xi, yi = xi.flatten(), yi.flatten()
        coords = np.vstack((yi, xi)) 
        
        zi = ndimage.map_coordinates(data, coords, order=1)
        return zi.reshape((nr, nt))

def ccf(data):
    return np.array([sp.correlate(p, np.concatenate((p,p))) for p in data])
    # return np.array([np.sum((sp.correlate(q, np.concatenate((p,p))) for p in data[n:n+1]),axis=0) for n,q in enumerate(data)])

def ccf_crossed(data):
    arr = []
    for p in data:
        for q in data:
            arr.append(sp.correlate(q, np.concatenate((p,p))))
    return arr
    # return np.array([sp.correlate(q, np.concatenate((p,p))) for q in data])

def ccf_angle(data,limit_edge_1,limit_edge_2):
    R, Q = data.shape[:2]
    ccf1d_rad = np.sum(data,axis=1)                 # Compute valuable limits of CCF
    ccf_limit_factor = 1000          # when to set limit? when value become factor times less than maximum

    ccf_max_norm = 1000.0

    ccf_rad_used_range_points = (ccf1d_rad>ccf1d_rad.max()/ccf_limit_factor) #array of points where condition is true
    ccf_rad_used_range = np.where(ccf_rad_used_range_points == ccf_rad_used_range_points.max()) # array of indices where condition is true

    ccf_rad_limit_1 = max(ccf_rad_used_range[0][0],limit_edge_1)
    ccf_rad_limit_2 = ccf_rad_used_range[0][-1]

    ccf_rad_limit_1 = limit_edge_1
    ccf_rad_limit_2 = limit_edge_2

    # print ccf1d_rad.argmax(), ccf_rad_limit_1, ccf_rad_limit_2

    ccf1d_ang = np.average(data[ccf_rad_limit_1:ccf_rad_limit_2],axis=0)

    ccf1d_ang /= ccf1d_ang.max()/ccf_max_norm

    return ccf1d_ang.max() - ccf1d_ang         # set 0 in endpoints

def ccf_rad(data):
    R, Q = data.shape[:2]
    ccf_max_norm = 1000.0

    ccf_rad_limit_1 = 40
    ccf_rad_limit_2 = R//2

    ccf1d_rad = np.log(np.average(data[ccf_rad_limit_1:ccf_rad_limit_2],axis=1))

    ccf1d_rad /= ccf1d_rad.max()/ccf_max_norm

    ccf1d_rad -= ndimage.gaussian_filter(ccf1d_rad, sigma=10)

    return ccf1d_rad.max() - ccf1d_rad 

def CalculateOmnilightIntencity(data,limit_edge_1):
    R = len(data)
    # Intencity ~ 1/r^2
    # so I_1 = c2 /((x_1+L_1)^2+ c1)
    # I_2 = c2 /((x_2+L_1)^2+ c1)
    #
    # c1 = (I_2*(x_2+L_1)^2 - I_1*(x_1+L_1)^2) / (I_1 - I_2)
    #
    # c2 = ( (x_2+L_1)^2 - (x_1+L_1)^2 ) * (I_1*I_2/(I_1 - I_2))
    x1_c = 0
    x2_c = 10 

    x_1 = limit_edge_1 + x1_c
    x_2 = limit_edge_1 + x2_c - 1
    I_1 = data[x1_c]
    I_2 = data[x2_c]

    c1 = (I_2*x_2**2 - I_1*x_1**2) / (I_1 - I_2)
    c2 = (x_2**2 - x_1**2) * (I_1*I_2) / (I_1 - I_2)

    omni_intencity = [c2/((x_1+i)**2 + c1) for i in range(R)]

    return omni_intencity


def CalcPointAsAverageDiagonalCenter(data):  # Predict by symmetry, slice image near center and  average symmetry point

    Y_size, X_size = data.shape[:2]

    M_avg_list = [CenterBySimmetry(np.diag(data,X_size//2-Y_size//2+ i),center_off = i) for i in xrange( -2, 3 )]               # Center_off used when center of geometry are 
    O_avg_list = [CenterBySimmetry(np.diag(np.fliplr(data),X_size//2-Y_size//2 + i),center_off = i) for i in xrange( -2, 3 )]   # not in the central position of array, especially for diagonals. 
                                                                                                                                # YSize> Xsize so center moves vertically. Add i = Diag num.
    M_avg = np.average(M_avg_list)  # average value of cross between oppos and main diagonals equals minus Oppos diagonal number
    O_avg = np.average(O_avg_list)  # Equals minus main diagonal number

    # M_avg = CenterBySimmetry(np.diag(data,X_size//2-Y_size//2 - O_avg),center_off = -O_avg)
    # O_avg = CenterBySimmetry(np.diag(np.fliplr(data),X_size//2-Y_size//2 - M_avg),center_off = -M_avg)

    Y_dx = ( M_avg + O_avg )/2.0    # -1(Main Num + Opp num)/2
    X_dx = ( M_avg - O_avg )/2.0    # (Opp num - main num)

    return (X_dx,Y_dx)

def CalcPointAsAverageCrossCenter(data):  # Predict by symmetry, slice image near center and get average symmetry point

    Y_size, X_size = data.shape[:2]

    Y_avg_list = [CenterBySimmetry(data[:,X_size//2+i]) for i in xrange( -20, 20 )]
    Y_avg = np.average(Y_avg_list)

    X_avg_list = [CenterBySimmetry(data[Y_size//2+i,:]) for i in xrange( -20, 20 )]
    X_avg = np.average(X_avg_list)

    return (X_avg,Y_avg)

def CenterBySimmetry(vect, center_off =0): # Vector supposed to have maximin in center plus center_off
    R = len(vect)
    v_slice_left = vect[R//2::-1]    #First half oriented from Max to Min values
    v_slice_right = vect[R//2:]       #Second half oriented the same way

    values = {}
    for i in range(-10,10):
        left_shift = math.ceil((i+center_off)/2.0)       # it count as 00;01;11;12;22;23;33 and so on; Use to improve accuracy
        right_shift = math.floor((i+center_off)/2.0)

        v_slice_left = vect[R//4+left_shift:R//2+left_shift]   #First half oriented from Max to Min values
        v_slice_right = vect[R*3//4+right_shift:R//2+right_shift:-1]

        diff_vect = np.power(np.abs(v_slice_left - v_slice_right),2)

        slice_1 = len(v_slice_left) - np.argmax(v_slice_left)
        slice_r = len(v_slice_right) - np.argmax(v_slice_right)
        sl = (slice_1+slice_r)//2

        values[i] = np.sum(diff_vect[:-sl])

    return min(values, key=values.get)                                                   # Absolute position of symmetry center

def DiffValue(vect,center):
    R = len(vect)
    v_slice_left = vect[R//4+center:R*3//4+center].copy()    #First half oriented from Max to Min values
    v_slice_right = vect[R*3//4+center:R//4+center:-1].copy()
    # slice_1 = len(v_slice_left) - np.argmax(v_slice_left)
    # slice_r = len(v_slice_right) - np.argmax(v_slice_right)
    # sl = (slice_1+slice_r)//2
    diff_vect = np.power(np.abs(v_slice_left - v_slice_right),2)
    return np.sum(diff_vect[:])

def CalcSymmetryValueWithDiagonals(image_data,origin):
    X, Y = origin
    Y_size,X_size = image_data.shape[:2]
    v_slice = image_data[:,X]
    v_slice_center = Y - Y_size//2
    h_slice = image_data[Y,:]
    h_slice_center = X - X_size//2

    main_diag_number = X-Y
    oppos_diag_number = X_size-X-Y

    main_diag_slice = np.diag(image_data,main_diag_number)
    oppos_diag_slice = np.diag(np.fliplr(image_data),oppos_diag_number)

    main_diag_center = (Y+X) - (Y_size//2+X_size//2)
    oppos_diag_center = (Y-X) - (Y_size//2-X_size//2)
    # print v_slice_center,h_slice_center, main_diag_center, oppos_diag_center
    return  DiffValue(v_slice,v_slice_center) + DiffValue(h_slice,h_slice_center) + \
            DiffValue(main_diag_slice,main_diag_center) + DiffValue(oppos_diag_slice,oppos_diag_center)

def CalcPointWithSymmetryMaximum(data, offset = 10,zoom = 1.0):
    worked_data = data
    if zoom != 1.0:
        worked_data = ndimage.zoom(data, zoom, order=3)

    Y,X = data.shape[:2]
    Y_size,X_size = worked_data.shape[:2]

    step = 1
    n_steps_side = zoom*offset//step 

    l_step = -1*n_steps_side*step
    r_step = (n_steps_side+1)*step

    limit_x_1 = l_step
    limit_x_2 = r_step
    limit_y_1 = l_step
    limit_y_2 = r_step

    mtf = 1<<16     #Number BIG enough, more than maximum value
    ox = oy = None
    
    dx = np.arange(limit_x_1,limit_x_2,step)
    dy = np.arange(limit_y_1,limit_y_2,step)
    for x in dx:
        for y in dy:
            origin = ((X//2)*zoom+x,(Y//2)*zoom+y)
            new_mtf = CalcSymmetryValueWithDiagonals(worked_data,origin)
            if new_mtf < mtf:
                ox = x
                oy = y
                mtf = new_mtf

    # if X%2 == 1: ox+=zoom//2   #If Size if odd: center of data and worked_data are not the same
    # if Y%2 == 1: oy+=zoom//2

    return (ox/zoom,oy/zoom)


def PredictOrigin(data):
    smooth_data = np.log(ndimage.gaussian_filter(data, sigma=3)+1)
    C_1 = CalcPointAsAverageCrossCenter(smooth_data)
    C_2 = CalcPointAsAverageDiagonalCenter(smooth_data)
    C_Result = ((C_1[0] + C_2[0])/2.0,(C_1[1] + C_2[1])/2.0)
    return C_2

def GetLimitEdge(data,origin_shift=(0,0)):
    Y,X = data.shape[:2]
    max_point = np.unravel_index(data.argmax(),data.shape)  #Position of point with maximum value 
    center = (X//2, Y//2)
    blind_r = np.linalg.norm(np.subtract(max_point,center)) #distance between this point and the center of image
    blind_r = 40
    limit_edge = blind_r + np.linalg.norm(origin_shift)  #enough range to cover blind zone in the center
    return limit_edge

def GetEdgesForPolarByIntensity(polar_data,limit_edge):
    R,Q = polar_data.shape[:2]
    max_for_radius = np.amax(polar_data,axis=1)
    max_image_intensity = np.amax(polar_data)
    max_intensity = max_image_intensity/2.0
    min_intensity = max_image_intensity/10.0
    limit_inner = np.argmax(max_for_radius[limit_edge:]<max_intensity) + limit_edge
    limit_outer = R - np.argmax(max_for_radius[:limit_edge:-1]>min_intensity) + limit_edge
    # print limit_inner,limit_outer,limit_edge
    return limit_inner,limit_outer
