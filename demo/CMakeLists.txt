set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5Widgets REQUIRED)
find_package(Qt5OpenGL  REQUIRED)
find_package(OpenGL     REQUIRED)

include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${Qt5OpenGL_INCLUDE_DIRS})


file(GLOB src "*.cc")
add_executable(demo ${src})
target_link_libraries(demo bitplanes_core bitplanes_utils
  Qt5::Widgets Qt5::OpenGL ${OPENGL_LIBRARIES} ${MY_LIBRARIES})


