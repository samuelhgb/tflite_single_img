# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

# Declares and names the project.

project("tflite")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

#opencv
set(OpenCV_STATIC ON)
set(OpenCV_DIR $ENV{OPENCV_ANDROID}/sdk/native/jni)
find_package(OpenCV REQUIRED)


# Specify where to find the header files for TF Lite C++
set( INCLUDE_DIRS
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/include/flatbuffers/include)
include_directories(${INCLUDE_DIRS})

add_library( tflite_c SHARED IMPORTED )
set_target_properties( tflite_c PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/tf-lite-api/generated-libs/${ANDROID_ABI}/libtensorflowlite.so )

add_library( # Sets the name of the library.
        tflite

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)
find_library(android-lib android) # for AssetManager functionality

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
        tflite

        # Links the target library to the log library
        # included in the NDK.
        ${OpenCV_LIBS}
        ${log-lib}
        ${android-lib} tflite_c)