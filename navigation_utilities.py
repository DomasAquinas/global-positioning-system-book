"""
Ports of Navigation Utilities functions from gpstextbook.com supplementary data
for use with Python and NumPy

Some corrections are made
"""

import numpy as np


# Useful constants
WGS84_EQ_EARTH_RADIUS_M = 6378137.0
WGS84_FLATTENING = 1.0/298.257223563
ECC_SQ = (2 - WGS84_FLATTENING) * WGS84_FLATTENING
DEG_FROM_RAD = 180.0 / np.pi


def wgs_lla_from_xyz(xyz_pos_m):
    """
    Obtain longitude, latitude, and altitude (LLA) coordinates from x, y, z
    """

    # Calculate longitude from x and ywgs84_longitude
    if xyz_pos_m[0] == 0.0 and xyz_pos_m[1] == 0.0:
        wgs84_longitude = 0.0
    else:
        wgs84_longitude = np.arctan2(xyz_pos_m[1], xyz_pos_m[0]) * DEG_FROM_RAD

    # Check for the center of the earth case
    if all([coord == 0.0 for coord in xyz_pos_m]):
        raise ValueError('WGS x,y,z at center of Earth')

    # Solve for the coordinates iteratively
    else:

        # Calculate the XY distance from the origin and the range
        xy_dist_sq = np.sum(np.square(xyz_pos_m[0:2]))
        xy_dist = np.sqrt(xy_dist_sq)
        range = np.sqrt(xy_dist_sq + np.square(xyz_pos_m[2]))

        # Use a spherical Earth model for initial guesses
        temp_lat = np.arctan(xyz_pos_m[2] / xy_dist)
        temp_alt = range - WGS84_EQ_EARTH_RADIUS_M

        # Initialize errors
        xy_dist_error = 1000.0
        z_error = 1000.0

        # Iterate using Newton's method
        while np.abs(xy_dist_error) > 1.0e-6 or np.abs(z_error) > 1.0e-6:

            # Get the sine and cosine of the latitude angle
            lat_sin = np.sin(temp_lat)
            lat_cos = np.cos(temp_lat)

            # Calculate the prime vertical length
            pv_len_denom_sq = 1 - ECC_SQ*(lat_sin**2)
            pv_len = WGS84_EQ_EARTH_RADIUS_M/np.sqrt(pv_len_denom_sq)

            # Determine the derivative of the prime vertical
            dpvlen_dlat = pv_len / pv_len_denom_sq*lat_sin*lat_cos

            # Calculate the xy position and z position errors
            xy_dist_error = (pv_len + temp_alt)*lat_cos - xy_dist
            z_error = (pv_len*(1 - ECC_SQ) + temp_alt)*lat_sin - xyz_pos_m[2]

            # Calculate the Jacobian matrix of the error function
            # [dxyerror_dlat  dxyerror_dalt]
            # [dzerror_dlat   dzerror_dalt ]
            j11 = dpvlen_dlat*lat_cos - (pv_len + temp_alt)*lat_sin
            j12 = lat_cos
            j21 = (1 - ECC_SQ)*(dpvlen_dlat*lat_sin + pv_len*lat_cos)
            j22 = lat_sin
            jacobian = np.array([[j11, j12],
                                 [j21, j22]])

            # Multiply the error vector by the inverse Jacobian
            adjustments = np.matmul(
                np.linalg.inv(jacobian),
                np.array([[xy_dist_error], [z_error]])
            )

            # Obtain new guesses from the adjustments
            temp_lat = temp_lat - adjustments[0][0]
            temp_alt = temp_alt - adjustments[1][0]

    return wgs84_longitude, temp_lat * DEG_FROM_RAD, temp_alt


def degree_min_sec_from_decimal(dec_degrees):
    """
    Convert a decimal number of degrees into degree, arc minute, arc second
    representation.
    """
    degrees = np.fix(dec_degrees)
    dec_arc_min = np.abs((dec_degrees - degrees) * 60)
    arc_min = np.fix(dec_arc_min)
    arc_sec = np.fix((dec_arc_min - arc_min) * 60)
    return degrees, arc_min, arc_sec


def get_rot_matrix(angle, axis):
    """
    Obtain a rotation matrix for a given angle around a specified axis
    """

    # Start from a 3x3 identity matrix
    rot_mat = np.eye(3)

    # Obtain sine and cosine values
    sine_ang = np.sin(angle / DEG_FROM_RAD)
    cosine_ang = np.cos(angle / DEG_FROM_RAD)

    # Fill the matrix according to the specified axis
    if axis == 1:
        rot_mat[1][1] = cosine_ang
        rot_mat[1][2] = sine_ang
        rot_mat[2][1] = -sine_ang
        rot_mat[2][2] = cosine_ang

    elif axis == 2:
        rot_mat[0][0] = cosine_ang
        rot_mat[0][2] = -sine_ang
        rot_mat[2][0] = sine_ang
        rot_mat[2][2] = cosine_ang

    elif axis == 3:
        rot_mat[0][0] = cosine_ang
        rot_mat[0][1] = sine_ang
        rot_mat[1][0] = -sine_ang
        rot_mat[1][1] = cosine_ang

    else:
        raise ValueError("Axis must be 1, 2, or 3")

    return rot_mat


def enu_from_xyz(vec_xyz, lon, lat):
    """
    Perform rotations to express a WGS 84 ECEF Cartesian position vector in
    the East-North-Up frame with origin at the given longitude and latitude
    """

    rot_about_z = get_rot_matrix(90.0+lon, 3)
    rot_about_x = get_rot_matrix(90.0-lat, 1)

    vec_enu = np.matmul(
        np.matmul(rot_about_x, rot_about_z),
        np.transpose(vec_xyz)
    )

    return vec_enu
