
file(GLOB SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
  ./*.cpp)

include_directories(../)
add_executable(backface WIN32 ./backface.rc ${SOURCES})
set_property(TARGET backface PROPERTY CXX_STANDARD 11)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:10000000")
target_link_libraries(backface goengine)
#set_source_files_properties(your.ico your.rc2 PROPERTIES LANGUAGE RC).
