

add_subdirectory(engine)
if (NOT WIN32)
add_subdirectory(nblapp)
add_subdirectory(model)
endif()
add_subdirectory(simplemodel)
add_subdirectory(leela)
add_subdirectory(demo)
if (WIN32)
add_subdirectory(backface)
endif()

