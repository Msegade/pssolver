find_package (GTest REQUIRED)

include_directories (${PROJECT_SOURCE_DIR}/src ${GTEST_INCLUDE_DIRS})                      

add_executable ( vectortest vectortest.cpp )
target_link_libraries (vectortest LINK_PUBLIC ${GTEST_BOTH_LIBRARIES} pssolver)

add_test ( VectorTest ${EXECUTABLE_OUTPUT_PATH}/vectortest)
