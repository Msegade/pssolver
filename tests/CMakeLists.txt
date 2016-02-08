
include_directories (${PROJECT_SOURCE_DIR}/src)                      

add_executable ( vectortest main.cpp vectortest.cpp )
target_link_libraries ( vectortest LINK_PUBLIC pssolver )

add_test ( VectorTest ${EXECUTABLE_OUTPUT_PATH}/vectortest)
