cmake_minimum_required(VERSION 3.20)

project(
    MedicalAIWorkstation
    VERSION 0.1.0
    DESCRIPTION "C++/Qt workstation for medical image viewing and validated AI inference"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime C++ distribution")
set(MAIW_CARDIAC_MRI_PACKAGE_DIR "" CACHE PATH "External cardiac MRI deployment package root")

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must point to an ONNX Runtime C++ distribution")
endif()

find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
)

find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime headers/library were not found under ONNXRUNTIME_ROOT")
endif()

add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
set_target_properties(
    ONNXRuntime::ONNXRuntime
    PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

find_package(nlohmann_json REQUIRED)

add_library(maiw-onnx-runtime
    src/onnx/OnnxRuntimeSession.cpp
)

target_include_directories(maiw-onnx-runtime
    PUBLIC
        include
)

target_link_libraries(maiw-onnx-runtime
    PUBLIC
        ONNXRuntime::ONNXRuntime
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(maiw-onnx-runtime
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

if(BUILD_TESTING)
    if(NOT MAIW_CARDIAC_MRI_PACKAGE_DIR)
        message(STATUS "MAIW_CARDIAC_MRI_PACKAGE_DIR is not set; cardiac ONNX golden parity test disabled")
    else()
        find_package(OpenSSL REQUIRED COMPONENTS Crypto)

        add_executable(test-cardiac-mri-onnx-golden-parity
            tests/cardiac/TestCardiacMriOnnxGoldenParity.cpp
            tests/support/NpyReader.cpp
            tests/support/Sha256.cpp
        )

        target_include_directories(test-cardiac-mri-onnx-golden-parity
            PRIVATE
                tests/support
        )

        target_compile_definitions(test-cardiac-mri-onnx-golden-parity
            PRIVATE
                MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
        )

        target_link_libraries(test-cardiac-mri-onnx-golden-parity
            PRIVATE
                maiw-onnx-runtime
                nlohmann_json::nlohmann_json
                OpenSSL::Crypto
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-cardiac-mri-onnx-golden-parity
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        set_target_properties(test-cardiac-mri-onnx-golden-parity
            PROPERTIES
                BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        )

        add_test(
            NAME cardiac-mri-onnx-golden-parity
            COMMAND test-cardiac-mri-onnx-golden-parity
        )
    endif()
endif()
