#! /usr/bin/env python
import audioop

import numpy as np
import roslib
import rospy
import operator
import collections


import sys
import cv2

import numpy as np

import actionlib
import kinova_msgs.msg
import std_msgs.msg
import geometry_msgs.msg
from realsense_camera import *
from mask_rcnn import *


import math

""" Global variable """
rs = RealsenseCamera()
mrcnn = MaskRCNN()

arm_joint_number = 0
finger_number = 0
prefix = 'NO_ROBOT_TYPE_DEFINED_'
finger_maxDist = 18.9/2/1000  # max distance for one finger
finger_maxTurn = 6800  # max thread rotation for one finger
currentCartesianCommand = [0.212322831154, -0.257197618484, 0.509646713734, 1.63771402836, 1.11316478252, 0.134094119072] # default home in unit mq

def cartesian_pose_client(position, orientation):
    action_address = '/' + prefix + 'driver/pose_action/tool_pose'
    client = actionlib.SimpleActionClient(action_address, kinova_msgs.msg.ArmPoseAction)
    client.wait_for_server()
    goal = kinova_msgs.msg.ArmPoseGoal()
    goal.pose.header = std_msgs.msg.Header(frame_id=(prefix + 'link_base'))
    goal.pose.pose.position = geometry_msgs.msg.Point(
        x=position[0], y=position[1], z=position[2])
    goal.pose.pose.orientation = geometry_msgs.msg.Quaternion(
        x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
    print goal

    client.send_goal(goal)

    if client.wait_for_result(rospy.Duration(10.0)):
        return client.get_result()
    else:
        client.cancel_all_goals()
        print('        the cartesian action timed-out')
        return None

def getcurrentCartesianCommand(prefix_):
    # wait to get current position
    topic_address = '/' + prefix_ + 'driver/out/cartesian_command'
    rospy.Subscriber(topic_address, kinova_msgs.msg.KinovaPose, setcurrentCartesianCommand)
    rospy.wait_for_message(topic_address, kinova_msgs.msg.KinovaPose)
    print
    'position listener obtained message for Cartesian pose. '

def setcurrentCartesianCommand(feedback):
    global currentCartesianCommand

    currentCartesianCommand_str_list = str(feedback).split("\n")

    for index in range(0,len(currentCartesianCommand_str_list)):
        temp_str=currentCartesianCommand_str_list[index].split(": ")
        currentCartesianCommand[index] = float(temp_str[1])

def kinova_robotTypeParser(kinova_robotType_):
    """ Argument kinova_robotType """
    global robot_category, robot_category_version, wrist_type, arm_joint_number, robot_mode, finger_number, prefix, finger_maxDist, finger_maxTurn
    robot_category = kinova_robotType_[0]
    robot_category_version = int(kinova_robotType_[1])
    wrist_type = kinova_robotType_[2]
    arm_joint_number = int(kinova_robotType_[3])
    robot_mode = kinova_robotType_[4]
    finger_number = int(kinova_robotType_[5])
    prefix = kinova_robotType_ + "_"
    finger_maxDist = 18.9/2/1000  # max distance for one finger in meter
    finger_maxTurn = 6800  # max thread turn for one finger


def EulerXYZ2Quaternion(EulerXYZ_):
    tx_, ty_, tz_ = EulerXYZ_[0:3]
    sx = math.sin(0.5 * tx_)
    cx = math.cos(0.5 * tx_)
    sy = math.sin(0.5 * ty_)
    cy = math.cos(0.5 * ty_)
    sz = math.sin(0.5 * tz_)
    cz = math.cos(0.5 * tz_)

    qx_ = sx * cy * cz + cx * sy * sz
    qy_ = -sx * cy * sz + cx * sy * cz
    qz_ = sx * sy * cz + cx * cy * sz
    qw_ = -sx * sy * sz + cx * cy * cz

    Q_ = [qx_, qy_, qz_, qw_]
    return Q_


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])


def Rz(theta):
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])

def unitParser(pose_value):
    global currentCartesianCommand
    quant = EulerXYZ2Quaternion(currentCartesianCommand[3:])

    position = pose_value
    R = quaternion_rotation_matrix(quant)
    #print quant
    #print R
    pose = R * np.matrix(pose_value).T
    position = np.transpose(pose) + currentCartesianCommand[0:3]
    # orientation_deg_list = list(map(math.degrees, currentCartesianCommand[3:]))
    # Z = orientation_deg_list[2]
    # Y = orientation_deg_list[1]
    # X = orientation_deg_list[0]
    #
    #
    # R = Rz(Z) * Ry(Y) * Rx(X)
    #
    #
    # VF = R * np.matrix(pose_value).T
    #
    # VF = np.transpose(VF)
    #
    # # V = pose_value
    #
    # # T1 = np.array([[np.cos(Z), -np.sin(Z),0],[np.sin(Z), np.cos(Z), 0],[0, 0, 1]])
    # # T2 = np.array([[np.cos(Y), 0, np.sin(Y)], [0, 1, 0], [-np.sin(Y), 0, np.cos(Y)]])
    # # T3 = np.array([[1, 0, 0], [0, np.cos(X), -np.sin(X)], [0, np.sin(X), np.cos(X)]])
    # # v1 = np.matmul(T1, V)
    # # v2 = np.matmul(T2, v1)
    # # VF = np.matmul(T3, v2)
    # position = VF + currentCartesianCommand[0:3]
    # print VF
    # print currentCartesianCommand[0:3]
    #print position
    position = np.squeeze(np.asarray(position))



    #
    # for i in range(0,3):
    #     position = VF[i] + currentCartesianCommand[i]
    #current = currentCartesianCommand[:3]

    position = np.append(position, quant)


    return position




if __name__ == '__main__':
    kinova_robotType = 'j2n6s300'
    kinova_robotTypeParser(kinova_robotType)
    rospy.init_node(prefix + 'pose_action_client')
    default_position = [0.41786, -0.26509, 0.061973, 0.47201, 0.52273, 0.50061, 0.50334]
    getcurrentCartesianCommand(prefix)
    counter = 0
    d = collections.deque(maxlen=3)
    try:  # This part is to move Kinova to default position
        pose = [float(n) for n in default_position]

        result = cartesian_pose_client(pose[:3], pose[3:])


        print('Cartesian reset!')
    except rospy.ROSInterruptException:
        print "program interrupted before completion"



    while True:  # End-effector frame Kinova control
        ret, bgr_frame, depth_frame, depth_intrin, color_intrin = rs.get_frame_stream()

        boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

        # Draw object mask
        bgr_frame = mrcnn.draw_object_mask(bgr_frame)

        # Show depth info of the objects
        bgr_frame, close_x, close_y, close_z, close_class = mrcnn.draw_object_info(bgr_frame, depth_frame, depth_intrin)

        dist = [0, 0, 0]
        d.append(close_class)
        #print(d)
        if d.count(d[0]) == len(d):
            #apply attractor here:




            dist = [close_x / 1000, -close_y / 1000,
                    close_z / 1000]  # X for left or right, y for up or down, z for forward or backward
            offset = [0.15,0,0.15]
            true_dist = map(operator.sub, dist, offset)

            position = unitParser(true_dist)
            pose = [float(n) for n in position]
            x_dirct = pose[0]
            y_dirct = pose[1]
            z_dirct = pose[2]
            print(d)
            print(z_dirct)






        cv2.imshow("Bgr frame", bgr_frame)
        print close_z / 1000

        key = cv2.waitKey(1)
        counter = counter + 1
        if key == 97 or key == 65:  # which is A or a on keyboard
            dist = [close_x / 1000, -close_y / 1000,
                    close_z / 1000]  # X for left or right, y for up or down, z for forward or backward
            offset = [0.1,0.08,0.15]
            true_dist = map(operator.sub, dist, offset)
            try:
                position = unitParser(true_dist)
                pose = [float(n) for n in position]
                result = cartesian_pose_client(pose[:3], pose[3:])
                print('Cartesian pose sent')

                # print currentCartesianCommand
                # print pose

            except rospy.ROSInterruptException:
                print "program interrupted before completion"

        if key == 82 or key == 114:  # which is A or a on keyboard
            try:  # This part is to move Kinova to default position
                pose = [float(n) for n in default_position]

                result = cartesian_pose_client(pose[:3], pose[3:])

                print('Cartesian reset!')
            except rospy.ROSInterruptException:
                print "program interrupted before completion"


        if key == 27:
            break

    rs.release()
    cv2.destroyAllWindows()

    ################################# Play Ground without Realsense ##########################################################

 # which is A or a on keyboard
 #    dist = [0, 0, 0]  # X for left or right, y for up or down, z for forward or backward
 #    try:
 #        getcurrentCartesianCommand(prefix)
 #        position = unitParser(dist)
 #        pose = [float(n) for n in position]
 #
 #        result = cartesian_pose_client(pose[:3], pose[3:])
 #
 #        print('Cartesian pose sent')
 #
 #            # print currentCartesianCommand
 #            # print pose
 #
 #    except rospy.ROSInterruptException:
 #        print "program interrupted before completion"









