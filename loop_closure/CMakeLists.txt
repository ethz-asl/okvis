cmake_minimum_required(VERSION 2.8.11)
project(loop_closure)

if(NOT DO_TIMING)
  add_definitions(-DDEACTIVATE_TIMERS)
  message(STATUS "Deactivating timers.")
endif()

# require Eigen
find_package( Eigen REQUIRED )
include_directories(${EIGEN_INCLUDE_DIR}) 

# build the library
add_library(${PROJECT_NAME} STATIC
  src/KeyframeForLoopDetection.cpp
  src/LoopClosureModule.cpp
  src/LoopClosureMethod.cpp
  src/LoopClosureParameters.cpp
  src/LoopFrameAndMatches.cpp

  include/okvis/KeyframeForLoopDetection.hpp
  include/okvis/LoopClosureModule.hpp
  include/okvis/LoopClosureMethod.hpp
  include/okvis/LoopClosureParameters.hpp
  include/okvis/LoopFrameAndMatches.hpp)

# and link it
target_link_libraries(${PROJECT_NAME}
  PUBLIC okvis_cv
  PRIVATE ${GLOG_LIBRARIES})

# installation if required
install(TARGETS ${PROJECT_NAME}
  EXPORT okvisTargets 
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
)
install(DIRECTORY include/ DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev FILES_MATCHING PATTERN "*.hpp")
