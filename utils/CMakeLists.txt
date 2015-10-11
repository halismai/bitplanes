file(GLOB src "*.cc")
add_library(bitplanes_utils ${LIBRARY_TYPE} ${src})
set_target_properties(bitplanes_utils PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS bitplanes_utils DESTINATION lib)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/utils"
  DESTINATION include/bitplanes/ FILES_MATCHING PATTERN "*.h")



