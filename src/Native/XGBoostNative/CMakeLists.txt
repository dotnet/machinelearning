project (XGBoostNative)

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_definitions(-DUSESSE)
endif()

find_package(xgboost REQUIRED)

if (XGBOOST_BUILD_STATIC_LIB)
  enable_language(CXX)
  # find again for those  cxx libraries.
  find_package(xgboost REQUIRED)
endif(XGBOOST_BUILD_STATIC_LIB)

set(SOURCES XGBoostWrapper.cpp)
add_library(XGBoostNative SHARED ${SOURCES})
target_link_libraries(XGBoostNative PRIVATE xgboost::xgboost)
install_library_and_symbols (XGBoostNative)




