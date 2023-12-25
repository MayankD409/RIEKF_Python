# -*- coding: utf-8 -*-

from scipy.linalg import null_space
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial.transform import Rotation
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from google.colab import drive
import os
import scipy.io as sio

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def so2(x):
    t = np.linalg.norm(x)

    if t < 0.0001:
        y = np.eye(2)
    else:
        y = 1/t * np.array([[np.sin(t), np.cos(t)-1],
                            [1-np.cos(t), np.sin(t)]])

    return y

def exponential_so3(x):
    # Assuming x is a column vector
    theta = np.linalg.norm(x)

    if theta == 0:
        result = np.eye(3)
    else:
        omega = x / theta
        skew_omega = np.array([[0, -omega[2], omega[1]],
                               [omega[2], 0, -omega[0]],
                               [-omega[1], omega[0], 0]])

        result = np.eye(3) + np.sin(theta) * skew_omega + (1 - np.cos(theta)) * np.dot(skew_omega, skew_omega)

    return result


def right_jacobian(delta):
    N = (delta.shape[0] - 3) // 3

    if delta.shape[0] == 3:
        # Function Mode 1
        theta = np.linalg.norm(delta)
        if theta < 1e-8:
            result = np.eye(3)
        else:
            skew_delta = skew(delta)
            result = np.eye(3) - (1 - np.cos(theta)) * skew_delta / theta**2 + (theta - np.sin(theta)) * np.dot(skew_delta, skew_delta) / theta**3

    else:
        # Function Mode 2
        result = np.zeros((3 + 3 * N, 3 + 3 * N))
        s_theta = delta[:3]
        for i in range(1, N + 2):  # Set diagonal to J_r(s_theta)
            result[3 * i - 3:3 * i, 3 * i - 3:3 * i] = right_jacobian(s_theta)
        for i in range(1, N + 1):
            s_temp = delta[3 * i:3 * i + 3]
            result[3 * i:3 * i + 3, :3] = K_r_x1x2(s_theta, s_temp)

    return result


def K_r_x1x2(s_theta, s_temp):
    # Implement the K_r_x1x2 function as needed
    # You might need to create or import this function separately
    # and adjust the code accordingly.
    pass

def K_right(theta1, theta2):
    return Kl(-theta1, -theta2)

def Kl(x1, x2):
    theta = np.linalg.norm(x1)

    if theta > 0.00001:
        skew_x1 = skew(x1)
        skew_x2 = skew(x2)

        Kl_value = 1/2 * skew_x2 + \
                   (theta - np.sin(theta)) / theta**3 * (skew_x1 @ skew_x2 + skew_x2 @ skew_x1 + skew_x1 @ skew_x2 @ skew_x1) + \
                   -(1 - theta**2/2 - np.cos(theta)) / theta**4 * (skew_x1 @ skew_x1 @ skew_x2 + skew_x2 @ skew_x1 @ skew_x1 - 3 * skew_x1 @ skew_x2 @ skew_x1) + \
                   -1/2 * ((1 - theta**2/2 - np.cos(theta)) / theta**4 - 3 * (theta - np.sin(theta) - theta**3/6) / theta**5) * (skew_x1 @ skew_x2 @ skew_x1 @ skew_x1 + skew_x1 @ skew_x1 @ skew_x2 @ skew_x1)
    else:
        Kl_value = 1/2 * skew(x2)

    return Kl_value

def logarithmic_so3(R, *args):
    if len(args) > 0:
        if np.linalg.norm(R - np.eye(3)) < 2 * np.finfo(float).eps:
            f = np.zeros((3, 1))
            return f

    phi = np.arccos(1/2 * (np.trace(R) - 1))

    if len(args) > 0:
        if abs(phi) < 1e-10:
            f = np.zeros((3, 1))
            return f

    if np.linalg.norm(R - np.eye(3)) > 0.00001:
        f = inversehat_so3(phi / (2 * np.sin(phi)) * (R - np.transpose(R)))
    else:
        f = np.array([0, 0, 0]).reshape((3, 1))

    return f

def inversehat_so3(xi):
    v = np.array([xi[2, 1], xi[0, 2], xi[1, 0]])
    return v


def SE23(EV):
    R = exponential_so3(EV[0:3])
    t1 = right_jacobian(-EV[0:3]) @ EV[3:6]
    t2 = right_jacobian(-EV[0:3]) @ EV[6:9]

    y = np.column_stack((np.column_stack((R, t1)), t2))
    y = np.vstack((y, np.array([0, 0, 0, 1, 0, 0, 0, 1])))

    return y

def z_noise_add(z_expectation, z_expectation0, OBSV_noise):
    T_steps = len(z_expectation)
    z_noise = [None] * T_steps

    N = z_expectation0['position'].shape[1] if hasattr(z_expectation0, 'position') else 0
    if N < 1:
        z0_rotation = np.array([])
        z0_position = np.array([])
    else:
        z0_rotation = np.zeros((3, 3 * N))
        z0_position = np.zeros((3, N))
        for j in range(N):
            V = np.random.multivariate_normal(np.zeros(6), OBSV_noise).reshape(-1, 1)
            normV = np.linalg.norm(V[:3])
            normvec = V[:3] / normV if normV != 0 else np.zeros(3)
            if normV >= 2 * np.pi:
                normV = normV % (2 * np.pi)
                if normV > np.pi:
                    normV = normV - 2 * np.pi
            V[:3] = normV * normvec
            z0_rotation[:, 3 * j:3 * (j + 1)] = exponential_so3(V[:3]) @ z_expectation0['rotation'][:, 3 * j:3 * (j + 1)]
            z0_position[:, j] = z_expectation0['position'][:, j] + V[3:]

    for i in range(T_steps):
        N = z_expectation[i]['position'].shape[1] if 'position' in z_expectation[i] else 0
        if N < 1:
            z_noise[i] = {'rotation': np.array([]), 'position': np.array([])}
        else:
            z_noise[i] = {'rotation': np.zeros((3, 3 * N)), 'position': np.zeros((3, N))}
            for j in range(N):
                V = np.random.multivariate_normal(np.zeros(6), OBSV_noise).reshape(-1, 1)
                z_noise[i]['rotation'][:, 3 * j:3 * (j + 1)] = exponential_so3(V[:3]) @ z_expectation[i]['rotation'][:, 3 * j:3 * (j + 1)]
                z_noise[i]['position'][:, j] = z_expectation[i]['position'][:, j] + V[3:]

    return z_noise, {'rotation': z0_rotation, 'position': z0_position}

def fo_unoise(U, ODOM_noise):
    """
    Add noise to a sequence of 3D poses using first-order integration in Lie algebra.

    Args:
        U: A list of dictionaries representing the initial sequence of poses.
           Each dictionary should have 'rotation' (quaternion) and 'position' (3D array) keys.
        ODOM_noise: The covariance matrix of the odometry noise.

    Returns:
        U_noise: A list of dictionaries representing the noisy sequence of poses.
    """

    T_steps = len(U)
    U_noise = []

    for i in range(T_steps):
        # Generate a noise vector in Lie algebra
        W = np.random.multivariate_normal(mean=np.zeros(6), cov=ODOM_noise)


        # Normalize the rotational noise component
        normW = np.linalg.norm(W[:3])
        normvec = W[:3] / normW

        if normW >= 2 * np.pi:
            normW = np.mod(normW, 2 * np.pi)
            if normW > np.pi:
                normW = normW - 2 * np.pi

        W[:3] = normW * normvec

        # Exponentiate the noise vector to obtain the corresponding transformation in SE(3)
        exp_W = exponential_so3(W[:3])

        # Add the noise to the current pose
        rotated_quaternion = Rotation.from_quat(U[i]['rotation']).apply(exp_W[:3])
        U_noise.append({'rotation': rotated_quaternion.tolist(),
                        'position': U[i]['position'] + W[3:]})

    return U_noise

def remove_observation(Xn, Pn, Index):
    N = round(Xn.shape[1] / 4 - 1)
    n = len(Index)

    if n < 1:
        X = Xn.copy()
        P = Pn.copy()
    else:
        X = np.zeros((3, 4 * n + 4))
        X[0:3, 0:3] = Xn[0:3, 0:3]
        X[0:3, 3 * n + 4] = Xn[0:3, 3 * N + 4]

        P = np.zeros((6 * n + 6, 6 * n + 6))
        P[0:3, 0:3] = Pn[0:3, 0:3]
        P[0:3, 3 * n + 4:3 * n + 6] = Pn[0:3, 3 * N + 4:3 * N + 6]
        P[3 * n + 4:3 * n + 6, 0:3] = Pn[3 * N + 4:3 * N + 6, 0:3]
        P[3 * n + 4:3 * n + 6, 3 * n + 4:3 * n + 6] = Pn[3 * N + 4:3 * N + 6, 3 * N + 4:3 * N + 6]

        for i in range(n):
            X[0:3, 3 * i + 1:3 * i + 3] = Xn[0:3, 3 * Index[i] + 1:3 * Index[i] + 3]
            X[0:3, 3 * n + 4 + i] = Xn[0:3, 3 * N + 4 + Index[i]]

            P[0:3, 3 * i + 1:3 * i + 3] = Pn[0:3, 3 * Index[i] + 1:3 * Index[i] + 3]
            P[0:3, 3 * n + 4 + 3 * i:3 * n + 6 + 3 * i] = Pn[0:3, 3 * N + 4 + 3 * Index[i]:3 * N + 3 * Index[i] + 6]
            P[3 * i + 1:3 * i + 3, 0:3] = Pn[3 * Index[i] + 1:3 * Index[i] + 3, 0:3]
            P[3 * n + 4 + 3 * i:3 * n + 6 + 3 * i, 0:3] = Pn[3 * N + 4 + 3 * Index[i]:3 * N + 3 * Index[i] + 6, 0:3]

            P[3 * n + 4:3 * n + 6, 3 * i + 1:3 * i + 3] = Pn[3 * N + 4:3 * N + 6, 3 * Index[i] + 1:3 * Index[i] + 3]
            P[3 * n + 4:3 * n + 6, 3 * n + 4 + 3 * i:3 * n + 6 + 3 * i] = Pn[3 * N + 4:3 * N + 6, 3 * N + 4 + 3 * Index[i]:3 * N + 3 * Index[i] + 6]
            P[3 * i + 1:3 * i + 3, 3 * n + 4:3 * n + 6] = Pn[3 * Index[i] + 1:3 * Index[i] + 3, 3 * N + 4:3 * N + 6]
            P[3 * n + 4 + 3 * i:3 * n + 6 + 3 * i, 3 * n + 4:3 * n + 6] = Pn[3 * N + 4 + 3 * Index[i]:3 * N + 3 * Index[i] + 6, 3 * N + 4:3 * N + 6]

            for j in range(n):
                P[3 * i + 1:3 * i + 3, 3 * j + 1:3 * j + 3] = Pn[3 * Index[i] + 1:3 * Index[i] + 3, 3 * Index[j] + 1:3 * Index[j] + 3]
                P[3 * n + 4 + 3 * i:3 * n + 6 + 3 * i, 3 * n + 4 + 3 * j:3 * n + 6 + 3 * j] = Pn[3 * N + 4 + 3 * Index[i]:3 * N + 6 + 3 * Index[i], 3 * N + 4 + 3 * Index[j]:3 * N + 6 + 3 * Index[j]]

    return X, P

def standard_posestate(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    if N1 == N2:
        N = round(N1 / 4)
    else:
        raise ValueError('The dimensions of the inputs do not match!')

    R1 = X1[:, 0:3]
    R2 = X2[:, 0:3]
    X = np.zeros((3, 4 * N))
    X[:, 0:3] = np.dot(R1, R2)

    for i in range(1, N):
        X[:, 3 * i:3 * i + 3] = np.dot(X1[:, 3 * i:3 * i + 3], X2[:, 3 * i:3 * i + 3])

    for i in range(1, N + 1):
        X[:, 3 * N + i - 1] = X2[:, 3 * N + i - 1] + X1[:, 3 * N + i - 1]

    return X

def standard_minus_posestate(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    if N1 == N2:
        N = round(N1 / 4)  # N is the number of features + 1
    else:
        raise ValueError('The dimensions of the inputs do not match in minus_posestate!')

    R1 = X1[:, 0:3]
    R2 = X2[:, 0:3]
    X = np.zeros((3, 4 * N))

    for i in range(1, N + 1):
        X[:, 3 * i - 2:3 * i] = np.dot(X1[:, 3 * i - 2:3 * i], X2[:, 3 * i - 2:3 * i].T)
        X[:, 3 * N + i - 1] = -X2[:, 3 * N + i - 1] + X1[:, 3 * N + i - 1]

    xi = np.zeros((6 * N, 1))

    for i in range(1, N + 1):
        xi[3 * i - 2:3 * i, 0] = logarithmic_so3(X[:, 3 * i - 2:3 * i])

        normV = np.linalg.norm(xi[3 * i - 2:3 * i, 0])
        if normV < 1.0e-20:
            xi[3 * i - 2:3 * i, 0] = np.zeros((3, 1))
        else:
            normvec = xi[3 * i - 2:3 * i, 0] / normV
            if normV >= 2 * np.pi:
                normV = np.mod(normV, 2 * np.pi)
                if normV > np.pi:
                    normV = normV - 2 * np.pi
            xi[3 * i - 2:3 * i, 0] = normV * normvec

        if np.linalg.norm(X[:, 3 * N + i - 1]) < 1.0e-20:
            xi[3 * N + 3 * i - 2:3 * N + 3 * i, 0] = np.zeros((3, 1))
        else:
            xi[3 * N + 3 * i - 2:3 * N + 3 * i, 0] = X[:, 3 * N + i - 1]

    return xi

def standard_exponential_posestateonential(xi):
    N = round(xi.shape[0] / 6) - 1
    X = np.zeros((3, 4 * N + 4))
    X[:, 0:3] = exponential_so3(xi[0:3, 0])
    X[:, 3 * N + 3:3 * N + 6] = xi[3 * N + 3:3 * N + 6, 0]

    for i in range(1, N + 1):
        X[:, 3 * i:3 * i + 3] = exponential_so3(xi[3 * i:3 * i + 3, 0])
        X[:, 3 * N + 3 + i - 1] = xi[3 * N + 3 + 3 * i - 3:3 * N + 3 + 3 * i, 0]

    return X

def plus_posestate(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    if N1 == N2:
        N = round(N1 / 4)
    else:
        raise Warning('The dimensions of the inputs do not match!')

    R1 = X1[:, 0:3]
    R2 = X2[:, 0:3]
    X = np.zeros((3, 4 * N))
    X[:, 0:3] = np.dot(R1, R2)

    for i in range(1, N):
        X[:, 3 * i:3 * i + 3] = np.dot(X1[:, 3 * i:3 * i + 3], X2[:, 3 * i:3 * i + 3])

    for i in range(1, N + 1):
        X[:, 3 * N + i - 1] = np.dot(R1, X2[:, 3 * N + i - 1]) + X1[:, 3 * N + i - 1]

    return X

def minus_posestate(X1, X2):
    N1 = X1.shape[1]
    N2 = X2.shape[1]

    if N1 == N2:
        N = round(N1 / 4)  # N is the num of features + 1
    else:
        raise Warning('The dimensions of the inputs do not match in minus_posestate!')

    R1 = X1[:, 0:3]
    R2 = X2[:, 0:3]
    X = np.zeros((3, 4 * N))

    for i in range(1, N + 1):
        X[:, 3 * i - 2:3 * i] = np.dot(X1[:, 3 * i - 2:3 * i], X2[:, 3 * i - 2:3 * i].T)
        X[:, 3 * N + i - 1] = -np.dot(np.dot(R1, R2.T), X2[:, 3 * N + i - 1]) + X1[:, 3 * N + i - 1]

    xi = np.zeros((6 * N, 1))

    for i in range(1, N + 1):
        xi[3 * i - 2:3 * i, 0] = logarithmic_so3(X[:, 3 * i - 2:3 * i])

        normV = np.linalg.norm(xi[3 * i - 2:3 * i, 0])
        if normV < 1.0e-20:
            xi[3 * i - 2:3 * i, 0] = np.zeros((3, 1))
        else:
            normvec = xi[3 * i - 2:3 * i, 0] / normV
            if normV >= 2 * np.pi:
                normV = np.mod(normV, 2 * np.pi)
                if normV > np.pi:
                    normV = normV - 2 * np.pi
            xi[3 * i - 2:3 * i, 0] = normV * normvec

        if np.linalg.norm(X[:, 3 * N + i - 1]) < 1.0e-20:
            xi[3 * N + 3 * i - 2:3 * N + 3 * i, 0] = np.zeros((3, 1))
        else:
            xi[3 * N + 3 * i - 2:3 * N + 3 * i, 0] = np.linalg.inv(right_jacobian(-xi[1:4, 0])) @ X[:, 3 * N + i - 1]

    return xi

def exponential_posestate(xi):
    N = round(xi.shape[0] / 6) - 1
    X = np.zeros((3, 4 * N + 4))
    X[:, 0:3] = exponential_so3(xi[0:3, 0])
    X[:, 3 * N + 3:3 * N + 6] = right_jacobian(-xi[0:3, 0]) @ xi[3 * N + 3:3 * N + 6, 0]

    for i in range(1, N + 1):
        X[:, 3 * i:3 * i + 3] = exponential_so3(xi[3 * i:3 * i + 3, 0])
        X[:, 3 * N + 3 + i - 1] = right_jacobian(-xi[0:3, 0]) @ xi[3 * N + 3 + 3 * i - 3:3 * N + 6 + 3 * i - 3, 0]

    return X

def Augment(X, P, z, omega):
    N1 = X.shape[1]
    N2 = P.shape[1]
    N1 = round(N1 / 4) - 1
    N2 = round(N2 / 6) - 1

    if N1 == N2:
        N = N1
    else:
        raise ValueError("The dimensions of the inputs do not match in function 'Augment'!")

    if 'position' not in z or not isinstance(z['position'], np.ndarray) or len(z['position'].shape) < 2:
        print("Warning: 'position' not found or has invalid shape in dictionary 'z' in function 'Augment'!")
        N_new = 0
    else:
        N_new = round(z['position'].shape[1])

    R = X[0:3, 0:3]

    # New features
    f1 = np.zeros((3, 3 * N_new))
    f2 = np.zeros((3, N_new))

    for i in range(N_new):
        f1[0:3, 3*i:3*i+3] = R @ z['rotation'][0:3, 3*i:3*i+3]
        f2[0:3, i] = X[0:3, 3*N + 4] + R @ z['position'][0:3, i]

    # New state
    X_new = np.concatenate([X[0:3, 0:3 * N + 3], f1, X[0:3, 3 * N + 4:4 * N + 4], f2], axis=1)

    # New covariance
    P = (P + P.T) / 2
    P11 = P[0:3 * N + 3, 0:3 * N + 3]
    P12 = P[0:3 * N + 3, 3 * N + 4:6 * N + 6]
    P21 = P12.T
    P22 = P[3 * N + 4:6 * N + 6, 3 * N + 4:6 * N + 6]

    M1 = np.zeros((3 * N_new, 3 * N + 3))  # This assumes the size of M1 should be (3 * N_new) x (3 * N + 3)
    M2 = np.zeros((3 * N_new, 3 * N + 3))
    Omega11 = np.zeros((3 * N_new, 3 * N_new))
    Omega12 = np.zeros((3 * N_new, 3 * N_new))
    Omega22 = np.zeros((3 * N_new, 3 * N_new))
    R_blk = np.zeros((3 * N_new, 3 * N_new))

    for i in range(N_new):
        M1[3 * i:3 * (i + 1), 3 * N + 3:3 * N + 6] = np.eye(3)
        M2[3 * i:3 * (i + 1), 3 * N + 3:3 * N + 6] = np.eye(3)
        Omega11[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = omega[0:3, 0:3]
        Omega12[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = omega[0:3, 3:6]
        Omega22[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = omega[3:6, 3:6]
        R_blk[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R

    Omega21 = Omega12.T
    M1_T = M1.T
    M2_T = M2.T
    print("Shapes for debugging:")
    print("P11:", P11.shape)
    print("M1.T:", M1.T.shape)
    print("P12:", P12.shape)
    print("M2.T:", M2.T.shape)
    print("M1 @ P11:", (M1 @ P11).shape)
    print("M1 @ P11 @ M1.T:", (M1 @ P11 @ M1.T).shape)
    print("R_blk @ Omega11 @ R_blk.T:", (R_blk @ Omega11 @ R_blk.T).shape)
    # Check dimensions before multiplication
    if M1.T.shape[1] != P11.shape[0]:
        raise ValueError("Dimensions of matrices are not compatible")

    # Calculate M1 @ P11
    M1_P11 = np.matmul(M1, P11)

    P_new = np.block([[P11, P11 @ M1.T, P12, P12 @ M2.T],
                  [M1 @ P11, M1 @ P11 @ M1.T + R_blk @ Omega11 @ R_blk.T, M1 @ P12, M1 @ P12 @ M2.T + R_blk @ Omega12 @ R_blk.T],
                  [P21, P21 @ M1.T, P22, P22 @ M2.T],
                  [M2 @ P21, M2 @ P21 @ M1.T + R_blk @ Omega21 @ R_blk.T, M2 @ P22, M2 @ P22 @ M2.T + R_blk @ Omega22 @ R_blk.T]])

    return X_new, P_new

def standard_augment(X, P, z, omega):
    N1 = X.shape[1]
    N2 = P.shape[1]
    N1 = round(N1 / 4) - 1
    N2 = round(N2 / 6) - 1

    if N1 == N2:
        N = N1
    else:
        raise ValueError("The dimensions of the inputs do not match in function 'Augment'!")

    N_new = round(z.position.shape[1])
    R = X[0:3, 0:3]

    # New features
    f1 = np.zeros((3, 3 * N_new))
    f2 = np.zeros((3, N_new))

    for i in range(N_new):
        f1[0:3, 3*i:3*i+3] = R @ z.rotation[0:3, 3*i:3*i+3]
        f2[0:3, i] = X[0:3, 3*N + 4] + R @ z.position[0:3, i]

    # New state
    X_new = np.concatenate([X[0:3, 0:3 * N + 3], f1, X[0:3, 3 * N + 4:4 * N + 4], f2], axis=1)

    # New covariance
    P = (P + P.T) / 2
    P11 = P[0:3 * N + 3, 0:3 * N + 3]
    P12 = P[0:3 * N + 3, 3 * N + 4:6 * N + 6]
    P21 = P12.T
    P22 = P[3 * N + 4:6 * N + 6, 3 * N + 4:6 * N + 6]

    M1 = np.zeros((3 * N_new, 3 * N + 3))
    M2 = np.zeros((3 * N_new, 3 * N + 3))
    M3 = np.zeros((3 * N_new, 3 * N + 3))
    Omega11 = np.zeros((3 * N_new, 3 * N_new))
    Omega12 = np.zeros((3 * N_new, 3 * N_new))
    Omega22 = np.zeros((3 * N_new, 3 * N_new))
    R_blk = np.zeros((3 * N_new, 3 * N_new))

    for i in range(N_new):
        M1[3 * i:3 * i + 3, 0:3] = np.eye(3)
        M2[3 * i:3 * i + 3, 0:3] = -skew(R @ z.position[0:3, i])
        M3[3 * i:3 * i + 3, 0:3] = np.eye(3)
        Omega11[3 * i:3 * i + 3, 3 * i:3 * i + 3] = omega[0:3, 0:3]
        Omega12[3 * i:3 * i + 3, 3 * i:3 * i + 3] = omega[0:3, 3:6]
        Omega22[3 * i:3 * i + 3, 3 * i:3 * i + 3] = omega[3:6, 3:6]
        R_blk[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R

    Omega21 = Omega12.T

    P_new = np.block([[P11, P11 @ M1.T, P12, P11 @ M2.T + P12 @ M3.T],
                      [M1 @ P11, M1 @ P11 @ M1.T + R_blk @ Omega11 @ R_blk.T, M1 @ P12, M1 @ P11 @ M2.T + M1 @ P12 @ M3.T + R_blk @ Omega12 @ R_blk.T],
                      [P21, P21 @ M1.T, P22, P21 @ M2.T + P22 @ M3.T],
                      [M2 @ P11 + M3 @ P21, M2 @ P11 @ M1.T + M3 @ P21 @ M1.T + R_blk @ Omega21 @ R_blk.T, M2 @ P12 + M3 @ P22, M2 @ P11 @ M2.T + M2 @ P12 @ M3.T + M3 @ P21 @ M2.T + M3 @ P22 @ M3.T + R_blk @ Omega22 @ R_blk.T]])

    P_new = (P_new + P_new.T) / 2

    return X_new, P_new

def standardEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise):
    T_steps = len(Index)
    Xn, Pn = standard_augment(X0, P0, z0, OBSV_noise)
    X_estimation = []

    for i in range(T_steps):
        N = round(Xn.shape[1] / 4 - 1)
        X_Prediction = np.zeros((3, 4 * N + 4))
        X_Prediction[0:3, 0:3] = Xn[0:3, 0:3] @ U_noise[i]['rotation']
        X_Prediction[0:3, 4:3*N+3] = Xn[0:3, 4:3*N+3]
        X_Prediction[0:3, 3*N+4] = Xn[0:3, 0:3] @ U_noise[i]['position'] + Xn[0:3, 3*N+4]
        X_Prediction[0:3, 3*N+5:4*N+4] = Xn[0:3, 3*N+5:4*N+4]

        F = np.eye(6*N+6)
        F[3*N+4:3*N+6, 0:3] = -skew(Xn[0:3, 0:3] @ U_noise[i]['position'])

        A = np.zeros((6, 6))
        A[0:3, 0:3] = Xn[0:3, 0:3]
        A[3:6, 3:6] = Xn[0:3, 0:3]
        C = np.zeros((6*N, 6))

        P_adX_W = np.block([[A @ ODOM_noise @ A.T, A @ ODOM_noise @ C.T],
                            [C @ ODOM_noise.T @ A.T, C @ ODOM_noise @ C.T]])

        Pn_Prediction = F @ Pn @ F.T + P_adX_W
        Pn_Prediction = (Pn_Prediction + Pn_Prediction.T) / 2

        N_ob = len(Index[i]['RemainIndex'])
        H = np.zeros((6*N_ob, 6*N+6))

        for j in range(N_ob):
            H[3*j:3*j+3, 0:3] = -X_Prediction[0:3, 0:3].T
            H[3*j:3*j+3, 3*Index[i]['RemainIndex'][j]:3*Index[i]['RemainIndex'][j]+3] = X_Prediction[0:3, 0:3].T

            H[3*N_ob+3*j:3*N_ob+3*j+3, 0:3] = X_Prediction[0:3, 0:3].T @ skew(X_Prediction[0:3, 3*N+4+Index[i]['RemainIndex'][j]] - X_Prediction[0:3, 3*N+4])
            H[3*N_ob+3*j:3*N_ob+3*j+3, 3*N+4:3*N+6] = -X_Prediction[0:3, 0:3].T
            H[3*N_ob+3*j:3*N_ob+3*j+3, 3*N+4+3*Index[i]['RemainIndex'][j]:3*N+6+3*Index[i]['RemainIndex'][j]] = X_Prediction[0:3, 0:3].T

        Omega = np.block([[OBSV_noise[0:3, 0:3] * np.eye(3)] * N_ob,
                          [OBSV_noise[3:6, 3:6] * np.eye(3)] * N_ob])
        S = H @ Pn_Prediction @ H.T + Omega
        K = Pn_Prediction @ H.T @ np.linalg.inv(S)

        Y = np.zeros((6*N_ob, 1))

        for j in range(N_ob):
            V = logarithmic_so3(z_noise[i]['rotation'][0:3, 3*j:3*j+3] @ X_Prediction[0:3, 3*Index[i]['RemainIndex'][j]+1:3*Index[i]['RemainIndex'][j]+3].T @ X_Prediction[0:3, 0:3])
            normV = np.linalg.norm(V[0:3, 0])

            if normV < 1.0e-20:
                V[0:3, 0] = np.zeros((3, 1))
            else:
                normvec = V[0:3, 0] / normV
                if normV >= 2 * np.pi:
                    normV = normV % (2 * np.pi)
                    if normV > np.pi:
                        normV = normV - 2 * np.pi
                V[0:3, 0] = normV * normvec

            Y[3*j:3*j+3, 0] = V
            Y[3*N_ob+3*j:3*N_ob+3*j+3, 0] = z_noise[i]['position'][0:3, j] - X_Prediction[0:3, 0:3].T @ (X_Prediction[0:3, 3*N+4+Index[i]['RemainIndex'][j]] - X_Prediction[0:3, 3*N+4])

        Xn = standard_posestate(standard_exponential_posestateonential(K @ Y), X_Prediction)
        Pn = (np.eye(6*N+6) - K @ H) @ Pn_Prediction

        if z_noise[i]['position'][:, N_ob:].shape[1] > 0.5:
            z_new = {'rotation': z_noise[i]['rotation'][0:3, 3*N_ob:],
                     'position': z_noise[i]['position'][0:3, N_ob:]}
            Xn, Pn = standard_augment(Xn, Pn, z_new, OBSV_noise)
            Pn = (Pn + Pn.T) / 2

        X_estimation.append({'H': H, 'S': S, 'K': K, 'Y': Y, 'state': Xn, 'cov': Pn})

    return X_estimation

def RIEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise):
    T_steps = len(Index)
    Xn, Pn = Augment(X0, P0, z0, OBSV_noise)

    X_estimation = []

    for i in range(T_steps):
        [Xn, Pn] = remove_observation(Xn, Pn, Index[i].RemainIndex)
        N = round(Xn.shape[1] / 4 - 1)

        # Prediction
        X_Prediction = np.zeros((3, 4 * N + 4))
        X_Prediction[0:3, 0:3] = np.dot(Xn[0:3, 0:3], U_noise[i].rotation)
        X_Prediction[0:3, 4:3 * N + 3] = Xn[0:3, 4:3 * N + 3]
        X_Prediction[0:3, 3 * N + 4] = np.dot(Xn[0:3, 0:3], U_noise[i].position) + Xn[0:3, 3 * N + 4]
        X_Prediction[0:3, 3 * N + 5:4 * N + 4] = Xn[0:3, 3 * N + 5:4 * N + 4]

        A = np.zeros((6, 6))
        A[0:3, 0:3] = Xn[0:3, 0:3]
        A[3:6, 0:3] = skew(Xn[0:3, 3 * N + 4] + np.dot(Xn[0:3, 0:3], U_noise[i].position)) @ Xn[0:3, 0:3]
        A[3:6, 3:6] = Xn[0:3, 0:3]

        C = np.zeros((6 * N, 6))
        for j in range(N):
            C[3 * j:3 * j + 3, 0:3] = skew(Xn[0:3, 3 * N + 4 + j]) @ Xn[0:3, 0:3]

        P_adX_W = np.block([[A @ ODOM_noise @ A.T, A @ ODOM_noise @ C.T],
                            [C @ ODOM_noise.T @ A.T, C @ ODOM_noise @ C.T]])

        P_adX_W[3:6, :] = P_adX_W[6:9, :]
        P_adX_W[:, 3:6] = P_adX_W[:, 6:9]

        Pn_Prediction = Pn + P_adX_W
        Pn_Prediction = (Pn_Prediction + Pn_Prediction.T) / 2

        N_ob = len(Index[i].RemainIndex)
        H = np.zeros((6 * N_ob, 6 * N + 6))

        for j in range(N_ob):
            H[3 * j:3 * j + 3, 0:3] = -X_Prediction[0:3, 0:3].T
            H[3 * j:3 * j + 3, 3 * Index[i].RemainIndex[j] + 1:3 * Index[i].RemainIndex[j] + 4] = X_Prediction[0:3, 0:3].T

            H[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 3 * N + 4:3 * N + 6] = -X_Prediction[0:3, 0:3].T
            H[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 3 * N + 4 + 3 * Index[i].RemainIndex[j]:3 * N + 6 + 3 * Index[i].RemainIndex[j]] = X_Prediction[0:3, 0:3].T

        Omega = np.zeros((6 * N_ob, 6 * N_ob))
        for j in range(N_ob):
            Omega[3 * j:3 * j + 3, 3 * j:3 * j + 3] = OBSV_noise[0:3, 0:3]
            Omega[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3] = OBSV_noise[3:6, 3:6]

        S = H @ Pn_Prediction @ H.T + Omega
        K = Pn_Prediction @ H.T @ np.linalg.inv(S)

        Y = np.zeros((6 * N_ob, 1))
        for j in range(N_ob):
            V = logarithmic_so3(np.dot(z_noise[i].rotation[0:3, 3 * j:3 * j + 3],
                              np.dot(X_Prediction[0:3, 3 * Index[i].RemainIndex[j] + 1:3 * Index[i].RemainIndex[j] + 4].T,
                                     X_Prediction[0:3, 0:3].T)))
            normV = np.linalg.norm(V[0:3, 0])
            if normV < 1e-20:
                V[0:3, 0] = np.zeros((3, 1))
            else:
                normvec = V[0:3, 0] / normV
                if normV >= 2 * np.pi:
                    normV = normV % (2 * np.pi)
                    if normV > np.pi:
                        normV = normV - 2 * np.pi
                V[0:3, 0] = normV * normvec

            Y[3 * j:3 * j + 3, 0] = V
            Y[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 0] = z_noise[i].position[0:3, j] - np.dot(
                X_Prediction[0:3, 0:3].T, X_Prediction[0:3, 3 * N + 4 + Index[i].RemainIndex[j]] - X_Prediction[0:3, 3 * N + 4])

        Xn = plus_posestate(exponential_posestate(np.dot(K, Y)), X_Prediction)
        Pn = (np.eye(6 * N + 6) - K @ H) @ Pn_Prediction

        if z_noise[i].position[:, N_ob:].shape[1] > 0.5:
            z_new_rotation = z_noise[i].rotation[0:3, 3 * N_ob:]
            z_new_position = z_noise[i].position[0:3, N_ob:]
            z_new = {"rotation": z_new_rotation, "position": z_new_position}
            Xn, Pn = Augment(Xn, Pn, z_new, OBSV_noise)
            Pn = (Pn + Pn.T) / 2

        X_estimation.append({"H": H, "S": S, "K": K, "state": Xn, "cov": Pn})

    return X_estimation

def StdEKF_pose(X_Estimation, Xstate_gt, m):
    T_steps = len(X_Estimation)

    NEES_add = {
        'RobotRotation': np.zeros(T_steps),
        'RobotPosition': np.zeros(T_steps),
        'RobotPose': np.zeros(T_steps),
        'FeatureRotation': np.zeros(T_steps),
        'FeaturePosition': np.zeros(T_steps),
        'FeaturePose': np.zeros(T_steps),
        'Total': np.zeros(T_steps)
    }

    for i in range(T_steps):
        error = standard_minus_posestate(X_Estimation[i]['state'], Xstate_gt[i])
        N = round(error.shape[0] / 6) - 1

        NEES_add['RobotRotation'][i] = error[:3] @ np.linalg.solve(X_Estimation[i]['cov'][:3, :3], error[:3]) / (m * 3)
        NEES_add['RobotPosition'][i] = error[3 * N + 3:3 * N + 6] @ np.linalg.solve(X_Estimation[i]['cov'][3 * N + 3:3 * N + 6, 3 * N + 3:3 * N + 6], error[3 * N + 3:3 * N + 6]) / (m * 3)
        NEES_add['RobotPose'][i] = error[:3] @ np.linalg.solve(X_Estimation[i]['cov'][:3, :3], error[:3]) + \
                                   error[3 * N + 3:3 * N + 6] @ np.linalg.solve(X_Estimation[i]['cov'][3 * N + 3:3 * N + 6, 3 * N + 3:3 * N + 6], error[3 * N + 3:3 * N + 6]) / (m * 6)

        for j in range(N):
            idx = 3 * j + 3
            NEES_add['FeatureRotation'][i] += error[idx:idx + 3] @ np.linalg.solve(X_Estimation[i]['cov'][idx:idx + 3, idx:idx + 3], error[idx:idx + 3]) / (m * 3 * N)
            idx = 3 * N + 3 * j + 3
            NEES_add['FeaturePosition'][i] += error[idx:idx + 3] @ np.linalg.solve(X_Estimation[i]['cov'][idx:idx + 3, idx:idx + 3], error[idx:idx + 3]) / (m * 3 * N)

        NEES_add['FeaturePose'][i] = NEES_add['FeatureRotation'][i] + NEES_add['FeaturePosition'][i]

        NEES_add['Total'][i] = error @ np.linalg.solve(X_Estimation[i]['cov'], error) / (m * (6 * N + 6))

    return NEES_add

def NEES_poseadd(X_Estimation, Xstate_gt, m):
    T_steps = len(X_Estimation)

    NEES_add = {
        'RobotRotation': np.zeros(T_steps),
        'RobotPosition': np.zeros(T_steps),
        'RobotPose': np.zeros(T_steps),
        'FeatureRotation': np.zeros(T_steps),
        'FeaturePosition': np.zeros(T_steps),
        'FeaturePose': np.zeros(T_steps),
        'Total': np.zeros(T_steps)
    }

    for i in range(T_steps):
        error = minus_posestate(X_Estimation[i]['state'], Xstate_gt[i])
        N = round(error.shape[0] / 6) - 1

        NEES_add['RobotRotation'][i] = error[:3] @ np.linalg.solve(X_Estimation[i]['cov'][:3, :3], error[:3]) / (m * 3)
        NEES_add['RobotPosition'][i] = error[3 * N + 3:3 * N + 6] @ np.linalg.solve(X_Estimation[i]['cov'][3 * N + 3:3 * N + 6, 3 * N + 3:3 * N + 6], error[3 * N + 3:3 * N + 6]) / (m * 3)
        NEES_add['RobotPose'][i] = error[:3] @ np.linalg.solve(X_Estimation[i]['cov'][:3, :3], error[:3]) + \
                                   error[3 * N + 3:3 * N + 6] @ np.linalg.solve(X_Estimation[i]['cov'][3 * N + 3:3 * N + 6, 3 * N + 3:3 * N + 6], error[3 * N + 3:3 * N + 6]) / (m * 6)

        for j in range(N):
            idx = 3 * j + 3
            NEES_add['FeatureRotation'][i] += error[idx:idx + 3] @ np.linalg.solve(X_Estimation[i]['cov'][idx:idx + 3, idx:idx + 3], error[idx:idx + 3]) / (m * 3 * N)
            idx = 3 * N + 3 * j + 3
            NEES_add['FeaturePosition'][i] += error[idx:idx + 3] @ np.linalg.solve(X_Estimation[i]['cov'][idx:idx + 3, idx:idx + 3], error[idx:idx + 3]) / (m * 3 * N)

        NEES_add['FeaturePose'][i] = NEES_add['FeatureRotation'][i] + NEES_add['FeaturePosition'][i]

        NEES_add['Total'][i] = error @ np.linalg.solve(X_Estimation[i]['cov'], error) / (m * (6 * N + 6))

    return NEES_add

def MSEadd_std(X_Estimation, Xstate_gt, m):
    T_steps = len(X_Estimation)

    MSE_add = {
        'RobotRotation': np.zeros(T_steps),
        'RobotPosition': np.zeros(T_steps),
        'RobotPose': np.zeros(T_steps),
        'FeatureRotation': np.zeros(T_steps),
        'FeaturePosition': np.zeros(T_steps),
        'FeaturePose': np.zeros(T_steps),
        'Total': np.zeros(T_steps)
    }

    for i in range(T_steps):
        error = (standard_minus_posestate(X_Estimation[i]['state'], Xstate_gt[i]))**2 / m
        N = round(error.shape[0] / 6) - 1

        MSE_add['RobotRotation'][i] = np.sum(error[:3])
        MSE_add['RobotPosition'][i] = np.sum(error[3 * N + 3:3 * N + 6])
        MSE_add['RobotPose'][i] = MSE_add['RobotRotation'][i] + MSE_add['RobotPosition'][i]
        MSE_add['FeatureRotation'][i] = np.sum(error[3:3 * N + 3])
        MSE_add['FeaturePosition'][i] = np.sum(error[3 * N + 6:6 * N + 6])
        MSE_add['FeaturePose'][i] = MSE_add['FeatureRotation'][i] + MSE_add['FeaturePosition'][i]

    MSE_add['Total'] = MSE_add['RobotPose'] + MSE_add['FeaturePose']

    return MSE_add

def MSEadd(X_Estimation, Xstate_gt, m):
    T_steps = len(X_Estimation)

    MSE_add = {
        'RobotRotation': np.zeros(T_steps),
        'RobotPosition': np.zeros(T_steps),
        'RobotPose': np.zeros(T_steps),
        'FeatureRotation': np.zeros(T_steps),
        'FeaturePosition': np.zeros(T_steps),
        'FeaturePose': np.zeros(T_steps),
        'Total': np.zeros(T_steps)
    }

    for i in range(T_steps):
        error = (minus_posestate(X_Estimation[i]['state'], Xstate_gt[i]))**2 / m
        N = round(error.shape[0] / 6) - 1

        MSE_add['RobotRotation'][i] = np.sum(error[:3])
        MSE_add['RobotPosition'][i] = np.sum(error[3 * N + 3:3 * N + 6])
        MSE_add['RobotPose'][i] = MSE_add['RobotRotation'][i] + MSE_add['RobotPosition'][i]
        MSE_add['FeatureRotation'][i] = np.sum(error[3:3 * N + 3])
        MSE_add['FeaturePosition'][i] = np.sum(error[3 * N + 6:6 * N + 6])
        MSE_add['FeaturePose'][i] = MSE_add['FeatureRotation'][i] + MSE_add['FeaturePosition'][i]

    MSE_add['Total'] = MSE_add['RobotPose'] + MSE_add['FeaturePose']

    return MSE_add

def idealEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise, Xstate_gt):
    T_steps = len(Index)
    X_estimation = [None] * T_steps
    Xn, Pn = standard_augment(X0, P0, z0, OBSV_noise)

    for i in range(T_steps):
        N = round(Xn.shape[1] / 4 - 1)  # the num of features in the (i-1)-th state

        # Prediction
        X_Prediction = np.zeros((3, 4 * N + 4))
        X_Prediction[0:3, 0:3] = Xn[0:3, 0:3] @ U_noise[i]['rotation']
        X_Prediction[0:3, 3:3 * N + 3] = Xn[0:3, 3:3 * N + 3]
        X_Prediction[0:3, 3 * N + 4] = Xn[0:3, 0:3] @ U_noise[i]['position'] + Xn[0:3, 3 * N + 4]
        X_Prediction[0:3, 3 * N + 5:4 * N + 4] = Xn[0:3, 3 * N + 5:4 * N + 4]

        # Fn
        F = np.eye(6 * N + 6)
        if i == 0:
            F[3 * N + 4:3 * N + 6, 0:3] = -skew(Xstate_gt[i][0:3, 3 * N + 4])
        else:
            F[3 * N + 4:3 * N + 6, 0:3] = -skew(Xstate_gt[i][0:3, 3 * N + 4] - Xstate_gt[i - 1][0:3, 3 * N + 4])

        # Odometry noise is the first-order integration
        A = np.zeros((6, 6))
        if i == 0:
            A[0:3, 0:3] = np.eye(3)
            A[3:6, 3:6] = np.eye(3)
        else:
            A[0:3, 0:3] = Xstate_gt[i - 1][0:3, 0:3]
            A[3:6, 3:6] = Xstate_gt[i - 1][0:3, 0:3]

        C = np.zeros((6 * N, 6))
        P_adX_W = np.block([[A @ ODOM_noise @ A.T, A @ ODOM_noise @ C.T], [C @ ODOM_noise.T @ A.T, C @ ODOM_noise @ C.T]])

        # Prediction of cov of noise, P_{n+1|n}
        Pn_Prediction = F @ Pn @ F.T + P_adX_W
        Pn_Prediction = (Pn_Prediction + Pn_Prediction.T) / 2

        # H_{n+1}
        N_ob = len(Index[i]['RemainIndex'])
        H = np.zeros((6 * N_ob, 6 * N + 6))

        for j in range(N_ob):
            H[3 * j:3 * j + 3, 0:3] = -Xstate_gt[i][0:3, 0:3].T
            H[3 * j:3 * j + 3, 3 * Index[i]['RemainIndex'][j] + 1:3 * Index[i]['RemainIndex'][j] + 4] = Xstate_gt[i][0:3, 0:3].T

            H[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 0:3] = Xstate_gt[i][0:3, 0:3].T @ skew(
                Xstate_gt[i][0:3, 3 * N + 4 + Index[i]['RemainIndex'][j]] - X_Prediction[0:3, 3 * N + 4])
            H[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 3 * N + 4:3 * N + 6] = -Xstate_gt[i][0:3, 0:3].T
            H[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3,
            3 * N + 4 + 3 * Index[i]['RemainIndex'][j]:3 * N + 6 + 3 * Index[i]['RemainIndex'][j]] = \
            Xstate_gt[i][0:3, 0:3].T

        # Kalman Gain K_{n+1}
        Omega = np.zeros((6 * N_ob, 6 * N_ob))
        for j in range(N_ob):
            Omega[3 * j:3 * j + 3, 3 * j:3 * j + 3] = OBSV_noise[0:3, 0:3]
            Omega[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3] = OBSV_noise[3:6, 3:6]

        S = H @ Pn_Prediction @ H.T + Omega
        K = Pn_Prediction @ H.T @ np.linalg.inv(S)

        # Y_{n+1} = [y1_p1; y1_p2; ...; y2_pN]
        Y = np.zeros((6 * N_ob, 1))
        for j in range(N_ob):
            V = logarithmic_so3(z_noise[i]['rotation'][0:3, 3 * j:3 * j + 3] @ X_Prediction[0:3, 3 * Index[i]['RemainIndex'][j] + 1:3 * Index[i]['RemainIndex'][j] + 4].T @
                         X_Prediction[0:3, 0:3])
            normV = np.linalg.norm(V[0:3, 0])
            if normV < 1e-20:
                V[0:3, 0] = np.zeros((3, 1))
            else:
                normvec = V[0:3, 0] / normV
                if normV >= 2 * np.pi:
                    normV = np.mod(normV, 2 * np.pi)
                    if normV > np.pi:
                        normV = normV - 2 * np.pi
                V[0:3, 0] = normV * normvec
            Y[3 * j:3 * j + 3, 0] = V
            Y[3 * N_ob + 3 * j:3 * N_ob + 3 * j + 3, 0] = z_noise[i]['position'][0:3, j] - X_Prediction[0:3, 0:3].T @ \
                                                            (X_Prediction[0:3, 3 * N + 4 + Index[i]['RemainIndex'][j]] - X_Prediction[0:3, 3 * N + 4])

        Xn = standard_posestate(standard_exponential_posestateonential(K @ Y), X_Prediction)
        Pn = (np.eye(6 * N + 6) - K @ H) @ Pn_Prediction

        if z_noise[i]['position'][:, N_ob:].shape[1] > 0.5:
            z_new_rotation = z_noise[i]['rotation'][0:3, 3 * N_ob:]
            z_new_position = z_noise[i]['position'][0:3, N_ob:]
            z_new = {'rotation': z_new_rotation, 'position': z_new_position}
            Xn, Pn = standard_augment(Xn, Pn, z_new, OBSV_noise)
            Pn = (Pn + Pn.T) / 2

        X_estimation[i] = {'H': H, 'S': S, 'K': K, 'Y': Y, 'state': Xn, 'cov': Pn}

    return X_estimation

def combined_test(Xstate_gt, U, z_expectation, z_expectation0, X0, P0, Index, T_steps, m, ODOM_noise, OBSV_noise):
    # Initialize arrays for MSE and NEES
    MSE = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
           'FeatureRotation': np.zeros(T_steps[0]), 'FeaturePosition': np.zeros(T_steps[0])}

    NEES = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
            'RobotPose': np.zeros(T_steps[0]), 'FeatureRotation': np.zeros(T_steps[0]),
            'FeaturePosition': np.zeros(T_steps[0]), 'FeaturePose': np.zeros(T_steps[0])}

    # Initialize arrays for MSE_Std and NEES_Std
    MSE_Std = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
               'FeatureRotation': np.zeros(T_steps[0]), 'FeaturePosition': np.zeros(T_steps[0])}

    NEES_Std = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
                'RobotPose': np.zeros(T_steps[0]), 'FeatureRotation': np.zeros(T_steps[0]),
                'FeaturePosition': np.zeros(T_steps[0]), 'FeaturePose': np.zeros(T_steps[0])}

    # Initialize arrays for MSE_Ideal and NEES_Ideal
    MSE_Ideal = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
                 'FeatureRotation': np.zeros(T_steps[0]), 'FeaturePosition': np.zeros(T_steps[0])}

    NEES_Ideal = {'RobotRotation': np.zeros(T_steps[0]), 'RobotPosition': np.zeros(T_steps[0]),
                  'RobotPose': np.zeros(T_steps[0]), 'FeatureRotation': np.zeros(T_steps[0]),
                  'FeaturePosition': np.zeros(T_steps[0]), 'FeaturePose': np.zeros(T_steps[0])}

    for i in range(m):
        # Add noise to measurements and controls
        # U_noise = fo_unoise(U, ODOM_noise)
        U_noise = {'rotation': np.eye(3), 'position': np.zeros(3)}
        z_noise, z0 = z_noise_add(z_expectation, z_expectation0, OBSV_noise)

        # Run estimation algorithms
        X_Estimation = RIEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise)
        X_Estimation_std = standardEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise)
        X_Estimation_ideal = idealEKF_pose(X0, P0, z0, U_noise, z_noise, Index, ODOM_noise, OBSV_noise, Xstate_gt)

        # Update MSE and NEES
        MSE_add = MSEadd_std(X_Estimation, Xstate_gt, m)
        for key in MSE:
            MSE[key] += MSE_add[key]

        NEES_add = StdEKF_pose(X_Estimation, Xstate_gt, m)
        for key in NEES:
            NEES[key] += NEES_add[key]

        # Update MSE_Std and NEES_Std
        MSE_Std_add = MSEadd_std(X_Estimation_std, Xstate_gt, m)
        for key in MSE_Std:
            MSE_Std[key] += MSE_Std_add[key]

        NEES_Std_add = StdEKF_pose(X_Estimation_std, Xstate_gt, m)
        for key in NEES_Std:
            NEES_Std[key] += NEES_Std_add[key]

        # Update MSE_Ideal and NEES_Ideal
        MSE_Ideal_add = MSEadd_std(X_Estimation_ideal, Xstate_gt, m)
        for key in MSE_Ideal:
            MSE_Ideal[key] += MSE_Ideal_add[key]

        NEES_Ideal_add = StdEKF_pose(X_Estimation_ideal, Xstate_gt, m)
        for key in NEES_Ideal:
            NEES_Ideal[key] += NEES_Ideal_add[key]

    # Compute average values
    for key in MSE:
        MSE[key] /= m

    for key in NEES:
        NEES[key] /= m

    for key in MSE_Std:
        MSE_Std[key] /= m

    for key in NEES_Std:
        NEES_Std[key] /= m

    for key in MSE_Ideal:
        MSE_Ideal[key] /= m

    for key in NEES_Ideal:
        NEES_Ideal[key] /= m

    return MSE, NEES, MSE_Std, NEES_Std, MSE_Ideal, NEES_Ideal

# drive.mount('/content/drive')
data_directory = '/home/mayank/UMD/ENPM667/Project 1/python_code' # replace with your directory of data.mat file
os.chdir(data_directory)
data = sio.loadmat('data.mat')


# Load data
#data = np.load('data.npz')
Xstate_gt = data['Xstate_gt']
U = data['U']
z_expectation = data['z_expectation']
z_expectation0 = data['z_expectation0']
X0 = data['X0']
P0 = data['P0']
Index = data['Index']
T_steps = data['T_steps']
N = data['N']

m = 50  # the number of tests
P0 = np.zeros_like(P0)

# Settings
ODOM_noise = 1 * np.diag([0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2])
OBSV_noise = np.diag([1 * 0.1**2, 1 * 0.1**2, 1 * 0.1**2, 1 * 0.1**2, 1 * 0.1**2, 1 * 0.1**2])

# Statistics
MSE, NEES, MSE_Std, NEES_Std, MSE_Ideal, NEES_Ideal = combined_test(
    Xstate_gt, U, z_expectation, z_expectation0, X0, P0, Index, T_steps, m, ODOM_noise, OBSV_noise
)

# Calculate RMSE
RMSE = {
    'RobotRotation': np.sqrt(MSE['RobotRotation']),
    'RobotPosition': np.sqrt(MSE['RobotPosition']),
    'FeatureRotation': np.sqrt(MSE['FeatureRotation']),
    'FeaturePosition': np.sqrt(MSE['FeaturePosition'])
}

RMSE_Std = {
    'RobotRotation': np.sqrt(MSE_Std['RobotRotation']),
    'RobotPosition': np.sqrt(MSE_Std['RobotPosition']),
    'FeatureRotation': np.sqrt(MSE_Std['FeatureRotation']),
    'FeaturePosition': np.sqrt(MSE_Std['FeaturePosition'])
}

RMSE_Ideal = {
    'RobotRotation': np.sqrt(MSE_Ideal['RobotRotation']),
    'RobotPosition': np.sqrt(MSE_Ideal['RobotPosition']),
    'FeatureRotation': np.sqrt(MSE_Ideal['FeatureRotation']),
    'FeaturePosition': np.sqrt(MSE_Ideal['FeaturePosition'])
}

# Plot RMSE
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(range(1, T_steps + 1), RMSE['RobotRotation'], label='RI-EKF')
axes[0, 0].plot(range(1, T_steps + 1), RMSE_Std['RobotRotation'], label='Std-EKF')
axes[0, 0].plot(range(1, T_steps + 1), RMSE_Ideal['RobotRotation'], label='Ideal-EKF')
axes[0, 0].set_xlabel('Time Steps')
axes[0, 0].set_ylabel('RMSE')
axes[0, 0].set_title('RMSE for RobotRotation')
axes[0, 0].legend()

axes[0, 1].plot(range(1, T_steps + 1), RMSE['RobotPosition'], label='RI-EKF')
axes[0, 1].plot(range(1, T_steps + 1), RMSE_Std['RobotPosition'], label='Std-EKF')
axes[0, 1].plot(range(1, T_steps + 1), RMSE_Ideal['RobotPosition'], label='Ideal-EKF')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel('RMSE')
axes[0, 1].set_title('RMSE for RobotPosition')
axes[0, 1].legend()

axes[1, 0].plot(range(1, T_steps + 1), RMSE['FeatureRotation'], label='RI-EKF')
axes[1, 0].plot(range(1, T_steps + 1), RMSE_Std['FeatureRotation'], label='Std-EKF')
axes[1, 0].plot(range(1, T_steps + 1), RMSE_Ideal['FeatureRotation'], label='Ideal-EKF')
axes[1, 0].set_xlabel('Time Steps')
axes[1, 0].set_ylabel('RMSE')
axes[1, 0].set_title('RMSE for FeatureRotation')
axes[1, 0].legend()

axes[1, 1].plot(range(1, T_steps + 1), RMSE['FeaturePosition'], label='RI-EKF')
axes[1, 1].plot(range(1, T_steps + 1), RMSE_Std['FeaturePosition'], label='Std-EKF')
axes[1, 1].plot(range(1, T_steps + 1), RMSE_Ideal['FeaturePosition'], label='Ideal-EKF')
axes[1, 1].set_xlabel('Time Steps')
axes[1, 1].set_ylabel('RMSE')
axes[1, 1].set_title('RMSE for FeaturePosition')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Plot NEES
fig, axes = plt.subplots(3, 2, figsize=(12, 15))

axes[0, 0].plot(range(1, T_steps + 1), NEES['RobotPose'], label='RI-EKF', linewidth=0.7)
axes[0, 0].plot(range(1, T_steps + 1), NEES_Std['RobotPose'], label='Std-EKF', linewidth=0.7)
axes[0, 0].plot(range(1, T_steps + 1), NEES_Ideal['RobotPose'], label='Ideal-EKF', linewidth=0.7)
axes[0, 0].set_xlabel('Time Steps')
axes[0, 0].set_ylabel('NEES')
axes[0, 0].set_title('NEES for RobotPose')
axes[0, 0].legend()

axes[0, 1].plot(range(1, T_steps + 1), NEES['RobotRotation'], label='RI-EKF')
axes[0, 1].plot(range(1, T_steps + 1), NEES_Std['RobotRotation'], label='Std-EKF')
axes[0, 1].plot(range(1, T_steps + 1), NEES_Ideal['RobotRotation'], label='Ideal-EKF')
axes[0, 1].set_xlabel('Time Steps')
axes[0, 1].set_ylabel('NEES')
axes[0, 1].set_title('NEES for RobotRotation')
axes[0, 1].legend()

axes[1, 0].plot(range(1, T_steps + 1), NEES['RobotPosition'], label='RI-EKF', linewidth=0.7)
axes[1, 0].plot(range(1, T_steps + 1), NEES_Std['RobotPosition'], label='Std-EKF', linewidth=0.7)
axes[1, 0].plot(range(1, T_steps + 1), NEES_Ideal['RobotPosition'], label='Ideal-EKF', linewidth=0.7)
axes[1, 0].set_xlabel('Time Steps')
axes[1, 0].set_ylabel('NEES')
axes[1, 0].set_title('NEES for RobotPosition')
axes[1, 0].legend()

axes[1, 1].plot(range(1, T_steps + 1), NEES['FeaturePose'], label='RI-EKF', linewidth=1.5)
axes[1, 1].plot(range(1, T_steps + 1), NEES_Std['FeaturePose'], label='Std-EKF', linewidth=1.5)
axes[1, 1].plot(range(1, T_steps + 1), NEES_Ideal['FeaturePose'], label='Ideal-EKF', linewidth=1.5)
axes[1, 1].set_xlabel('Time Steps')
axes[1, 1].set_ylabel('NEES')
axes[1, 1].set_title('NEES for FeaturePose')
axes[1, 1].legend()

axes[2, 0].plot(range(1, T_steps + 1), NEES['FeatureRotation'], label='RI-EKF')
axes[2, 0].plot(range(1, T_steps + 1), NEES_Std['FeatureRotation'], label='Std-EKF')
axes[2, 0].plot(range(1, T_steps + 1), NEES_Ideal['FeatureRotation'], label='Ideal-EKF')
axes[2, 0].set_xlabel('Time Steps')
axes[2, 0].set_ylabel('NEES')
axes[2, 0].set_title('NEES for FeatureRotation')
axes[2, 0].legend()

axes[2, 1].plot(range(1, T_steps + 1), NEES['FeaturePosition'], label='RI-EKF')
axes[2, 1].plot(range(1, T_steps + 1), NEES_Std['FeaturePosition'], label='Std-EKF')
axes[2, 1].plot(range(1, T_steps + 1), NEES_Ideal['FeaturePosition'], label='Ideal-EKF')
axes[2, 1].set_xlabel('Time Steps')
axes[2, 1].set_ylabel('NEES')
axes[2, 1].set_title('NEES for FeaturePosition')
axes[2, 1].legend()

plt.tight_layout()
plt.show()

# Plot Experiment Map
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

Robot_Ori_x_gt = np.zeros((3, T_steps))
Robot_position_gt = np.zeros((3, T_steps))

for i in range(T_steps):
    Robot_Ori_x_gt[:, i] = Xstate_gt[i][:3, 0]
    Nn = Xstate_gt[i].shape[1] // 4
    Robot_position_gt[:, i] = Xstate_gt[i][:3, 3 * Nn]

Rp_x_gt = np.zeros((3, N))
Rp_y_gt = np.zeros((3, N))
Rp_z_gt = np.zeros((3, N))

for i in range(N):
    Rp_x_gt[:, i] = Xstate_gt[T_steps][:3, 3 * i + 1]
    Rp_y_gt[:, i] = Xstate_gt[T_steps][:3, 3 * i + 2]
    Rp_z_gt[:, i] = Xstate_gt[T_steps][:3, 3 * i + 3]

p_gt = Xstate_gt[T_steps][:3, 3 * N + 5:4 * N + 4]

ax.plot3D(Robot_position_gt[0, :], Robot_position_gt[1, :], Robot_position_gt[2, :], color=[0.6, 0.6, 0.6], linewidth=2)
ax.quiver(p_gt[0, :], p_gt[1, :], p_gt[2, :], Rp_x_gt[0, :], Rp_x_gt[1, :], Rp_x_gt[2, :], length=0.15, color=[0.6350, 0.0780, 0.1840], linewidth=2)
ax.quiver(p_gt[0, :], p_gt[1, :], p_gt[2, :], Rp_y_gt[0, :], Rp_y_gt[1, :], Rp_y_gt[2, :], length=0.15, color=[0, 0.4470, 0.7410], linewidth=2)
ax.quiver(p_gt[0, :], p_gt[1, :], p_gt[2, :], Rp_z_gt[0, :], Rp_z_gt[1, :], Rp_z_gt[2, :], length=0.15, color=[0.4660, 0.6740, 0.1880], linewidth=2)
ax.quiver(0, 0, 0, 1, 0, 0, length=0.5, color='k', arrow_length_ratio=0.05)
ax.scatter(0, 0, 0, marker='*', color='k', s=100, label='Origin')

ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Experiment Map')
ax.legend()

plt.savefig('Experiment_Map.png')
plt.show()