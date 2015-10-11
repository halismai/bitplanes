file(GLOB src "*.cc")
add_library(bitplanes_core ${LIBRARY_TYPE} ${src})
set_target_properties(bitplanes_core PROPERTIES LINKER_LANGUAGE CXX)

install(TARGETS bitplanes_core DESTINATION lib)
install(FILES "${PROJECT_BINARY_DIR}/core/bitplanes_config.h" DESTINATION
  include/bitplanes/core)
install(DIRECTORY "${PROJECT_SOURCE_DIR}/core"
  DESTINATION include/bitplanes/ FILES_MATCHING PATTERN "*.h")

