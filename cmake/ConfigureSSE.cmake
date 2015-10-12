find_package(SSE)

set(SSE_FLAGS "-march=native -mtune=native")
# there must be a better way of doing this, RTFM
macro(SetSSEDefs InstName)
  if(${${InstName}_FOUND})
    set(BITPLANES_HAVE_${InstName} 1)
    set(SSE_FLAGS "${SSE_FLAGS} ${${InstName}_FLAGS}")
  else()
    set(DIVO_HAVE_${InstName} 0)
  endif()
endmacro()

SetSSEDefs(SSE2)
SetSSEDefs(SSE3)
SetSSEDefs(SSSE3)
SetSSEDefs(SSE4_1)
SetSSEDefs(SSE4_2)
SetSSEDefs(AVX)
SetSSEDefs(AVX2)
SetSSEDefs(POPCNT)

message(STATUS "SSE: ${SSE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS}")

