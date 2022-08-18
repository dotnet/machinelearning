project (XGBoostNative)

if(NOT ${ARCHITECTURE} MATCHES "arm.*")
    add_definitions(-DUSESSE)
endif()

find_package(xgboost CONFIG REQUIRED)

set(SOURCES XGBoostWrapper.cpp)
add_library(XGBoostNative SHARED ${SOURCES})
install_library_and_symbols (XGBoostNative)




