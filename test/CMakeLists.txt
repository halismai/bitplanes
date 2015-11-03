file(GLOB src "test*.cc")

add_library(bitplanes_test ${LIBRARY_TYPE} slomo_data_loader.cc)
set_target_properties(bitplanes_test PROPERTIES LINKER_LANGUAGE CXX)

foreach(f ${src})
  get_filename_component(bname ${f} NAME_WE)
  add_executable(${bname} ${f})
  target_link_libraries(${bname} bitplanes_test bitplanes_core bitplanes_utils ${MY_LIBRARIES})
endforeach()

