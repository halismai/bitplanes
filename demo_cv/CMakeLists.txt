file(GLOB src "*.cc")
add_executable(main ${src})
target_link_libraries(main bitplanes_core bitplanes_utils ${MY_LIBRARIES})

