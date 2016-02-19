find_package (GTest REQUIRED)

include_directories (${PROJECT_SOURCE_DIR}/src ${GTEST_INCLUDE_DIRS})                      

file ( GLOB TEST_SOURCES *.cpp)

foreach (testsourcefile ${TEST_SOURCES} )
    get_filename_component (testbin ${testsourcefile} NAME_WE)
    cuda_add_executable ( ${testbin}  ${testsourcefile} )
    target_link_libraries (${testbin} ${GTEST_BOTH_LIBRARIES} pssolver)
    string ( TOUPPER ${testbin}  testname )
    add_test ( ${testname} ${EXECUTABLE_OUTPUT_PATH}/${testbin})
endforeach ( testsourcefile ${TEST_SOURCES} )
