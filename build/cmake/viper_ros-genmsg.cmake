# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "viper_ros: 0 messages, 1 services")

set(MSG_I_FLAGS "-Istd_msgs:/opt/ros/noetic/share/std_msgs/cmake/../msg;-Isensor_msgs:/opt/ros/noetic/share/sensor_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/noetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(viper_ros_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_custom_target(_viper_ros_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "viper_ros" "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" ""
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(viper_ros
  "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/viper_ros
)

### Generating Module File
_generate_module_cpp(viper_ros
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/viper_ros
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(viper_ros_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(viper_ros_generate_messages viper_ros_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_dependencies(viper_ros_generate_messages_cpp _viper_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(viper_ros_gencpp)
add_dependencies(viper_ros_gencpp viper_ros_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS viper_ros_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(viper_ros
  "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/viper_ros
)

### Generating Module File
_generate_module_eus(viper_ros
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/viper_ros
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(viper_ros_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(viper_ros_generate_messages viper_ros_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_dependencies(viper_ros_generate_messages_eus _viper_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(viper_ros_geneus)
add_dependencies(viper_ros_geneus viper_ros_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS viper_ros_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(viper_ros
  "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/viper_ros
)

### Generating Module File
_generate_module_lisp(viper_ros
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/viper_ros
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(viper_ros_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(viper_ros_generate_messages viper_ros_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_dependencies(viper_ros_generate_messages_lisp _viper_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(viper_ros_genlisp)
add_dependencies(viper_ros_genlisp viper_ros_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS viper_ros_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(viper_ros
  "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/viper_ros
)

### Generating Module File
_generate_module_nodejs(viper_ros
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/viper_ros
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(viper_ros_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(viper_ros_generate_messages viper_ros_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_dependencies(viper_ros_generate_messages_nodejs _viper_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(viper_ros_gennodejs)
add_dependencies(viper_ros_gennodejs viper_ros_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS viper_ros_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(viper_ros
  "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv"
  "${MSG_I_FLAGS}"
  ""
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/viper_ros
)

### Generating Module File
_generate_module_py(viper_ros
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/viper_ros
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(viper_ros_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(viper_ros_generate_messages viper_ros_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/kiyanoush/miscanthus_ws/src/viper_ros/srv/SavePointCloud.srv" NAME_WE)
add_dependencies(viper_ros_generate_messages_py _viper_ros_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(viper_ros_genpy)
add_dependencies(viper_ros_genpy viper_ros_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS viper_ros_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/viper_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/viper_ros
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(viper_ros_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(viper_ros_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/viper_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/viper_ros
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(viper_ros_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(viper_ros_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/viper_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/viper_ros
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(viper_ros_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(viper_ros_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/viper_ros)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/viper_ros
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(viper_ros_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(viper_ros_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/viper_ros)
  install(CODE "execute_process(COMMAND \"/usr/bin/python3\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/viper_ros\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/viper_ros
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(viper_ros_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(viper_ros_generate_messages_py sensor_msgs_generate_messages_py)
endif()
