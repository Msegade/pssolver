file ( GLOB_RECURSE hdr *.hpp )       
file ( GLOB_RECURSE cpp *.cpp )       

set ( SOURCES ${hdr} ${cpp} )   

add_library (pssolver SHARED ${SOURCES} )
