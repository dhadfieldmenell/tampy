# Install script for directory: /home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/michaelmcdonald/tampy/src/ros_utils/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros/msg" TYPE FILE FILES
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/PlanProb.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/PlanResult.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/PolicyUpdate.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/MotionPlanProblem.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/MotionPlanResult.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/HLProblem.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/HLPlanResult.msg"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/msg/PolicyPriorUpdate.msg"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros/srv" TYPE FILE FILES
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/srv/MotionPlan.srv"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/srv/PolicyAct.srv"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/srv/PolicyProb.srv"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/srv/Primitive.srv"
    "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/srv/QValue.srv"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros/cmake" TYPE FILE FILES "/home/michaelmcdonald/tampy/src/ros_utils/build/tamp_ros/catkin_generated/installspace/tamp_ros-msg-paths.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/michaelmcdonald/tampy/src/ros_utils/devel/include/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/roseus/ros" TYPE DIRECTORY FILES "/home/michaelmcdonald/tampy/src/ros_utils/devel/share/roseus/ros/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/common-lisp/ros" TYPE DIRECTORY FILES "/home/michaelmcdonald/tampy/src/ros_utils/devel/share/common-lisp/ros/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gennodejs/ros" TYPE DIRECTORY FILES "/home/michaelmcdonald/tampy/src/ros_utils/devel/share/gennodejs/ros/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  execute_process(COMMAND "/usr/bin/python" -m compileall "/home/michaelmcdonald/tampy/src/ros_utils/devel/lib/python2.7/dist-packages/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python2.7/dist-packages" TYPE DIRECTORY FILES "/home/michaelmcdonald/tampy/src/ros_utils/devel/lib/python2.7/dist-packages/tamp_ros")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "/home/michaelmcdonald/tampy/src/ros_utils/build/tamp_ros/catkin_generated/installspace/tamp_ros.pc")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros/cmake" TYPE FILE FILES "/home/michaelmcdonald/tampy/src/ros_utils/build/tamp_ros/catkin_generated/installspace/tamp_ros-msg-extras.cmake")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros/cmake" TYPE FILE FILES
    "/home/michaelmcdonald/tampy/src/ros_utils/build/tamp_ros/catkin_generated/installspace/tamp_rosConfig.cmake"
    "/home/michaelmcdonald/tampy/src/ros_utils/build/tamp_ros/catkin_generated/installspace/tamp_rosConfig-version.cmake"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/tamp_ros" TYPE FILE FILES "/home/michaelmcdonald/tampy/src/ros_utils/src/tamp_ros/package.xml")
endif()

