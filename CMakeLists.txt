cmake_minimum_required(VERSION 3.21)
project(privacy_filter_py LANGUAGES CXX)

# Static pf + ggml get linked into a single shared Python extension module, so
# the wheel is self-contained (no loose libggml.so to locate at runtime).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)  # build ggml statically

# We only need the pf library, not upstream's CLI/bench/test tree.
set(PF_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(PF_BUILD_TESTS OFF CACHE BOOL "" FORCE)

# CPU-only wheels: on Apple, ggml enables Metal by default, which would require
# shipping a .metallib alongside the extension. Keep the wheel self-contained.
# (Accelerate stays on — it is a system framework, nothing to bundle.)
if(APPLE)
    set(GGML_METAL OFF CACHE BOOL "" FORCE)
endif()

add_subdirectory(extern/privacy-filter.cpp)

# nanobind ships its CMake config with the pip package; scikit-build-core puts
# it on CMAKE_PREFIX_PATH.
find_package(Python 3.12 REQUIRED
    COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_core STABLE_ABI src/privacy_filter/binding.cpp)
# pf PUBLIC-links ggml, so it is pulled in transitively.
target_link_libraries(_core PRIVATE pf)

install(TARGETS _core LIBRARY DESTINATION privacy_filter)
