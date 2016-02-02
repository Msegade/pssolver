include_directories (${PROJECT_SOURCE_DIR}/src)                      
#file ( GLOB TEST_SOURCES *.cpp )                                 
#
#foreach ( testsourcefile ${TEST_SOURCES} )                       
#    get_filename_component ( testname ${testsourcefile} NAME_WE )
#    add_executable( ${testname} ${testsourcefile} )         
#    target_link_libraries ( ${testname} LINK_PUBLIC pssolver )               
#endforeach ( testsourcefile ${TEST_SOURCES} )                    

add_executable ( vectortest main.cpp vectortest.cpp )
target_link_libraries ( vectortest LINK_PUBLIC pssolver )
