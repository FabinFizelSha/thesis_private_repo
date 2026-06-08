cmake_minimum_required(VERSION 3.8)
project(rsg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

ament_python_install_package(rsg_preprocessing)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RsgFrame.msg"
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

install(PROGRAMS
  scripts/RSG_pre_processor
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  rsg_preprocessing
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
