set(CALIPER_CONTROLLERS_SOURCES
  controllers/CallpathSampleReportController.cpp
  controllers/CudaActivityProfileController.cpp
  controllers/CudaActivityReportController.cpp
  controllers/HatchetRegionProfileController.cpp
  controllers/HatchetSampleProfileController.cpp
  controllers/LoopReportController.cpp
  controllers/OpenMPReportController.cpp
  controllers/RuntimeReportController.cpp
  controllers/SpotController.cpp
  controllers/controllers.cpp
  controllers/util.cpp)

list(APPEND CALIPER_RUNTIME_SOURCES ${CALIPER_CONTROLLERS_SOURCES})
set(CALIPER_RUNTIME_SOURCES ${CALIPER_RUNTIME_SOURCES} PARENT_SCOPE)
