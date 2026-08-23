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
# Real ACDC data remains external so portable CI can configure without private medical datasets.
set(MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR "" CACHE PATH "External real ACDC training dataset root")
set(MAIW_CARDIAC_MRI_REAL_ED_PATH "" CACHE FILEPATH "Optional real standalone ED volume for cardiac preprocessing smoke testing")
set(MAIW_CARDIAC_MRI_REAL_ES_PATH "" CACHE FILEPATH "Optional real standalone ES volume for cardiac preprocessing smoke testing")
set(MAIW_QTVIEWERPRO_SOURCE_DIR "" CACHE PATH "Local qt-viewer-pro source directory")
option(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION "Enable local qt-viewer-pro core/IO source integration" OFF)

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNXRUNTIME_ROOT must point to an ONNX Runtime C++ distribution")
endif()

if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
    if(CMAKE_VERSION VERSION_LESS 3.28)
        message(FATAL_ERROR "MAIW_ENABLE_QTVIEWERPRO_INTEGRATION is a temporary development-only source integration that requires CMake 3.28 or newer. The upstream qt-viewer-pro build registers tests unconditionally and does not yet provide install/export packaging, so the workstation must use CMake 3.28 set_tests_properties(DIRECTORY) support to disable those upstream tests cleanly.")
    endif()
    if(NOT MAIW_QTVIEWERPRO_SOURCE_DIR)
        message(FATAL_ERROR "MAIW_QTVIEWERPRO_SOURCE_DIR must point to the qt-viewer-pro source tree when MAIW_ENABLE_QTVIEWERPRO_INTEGRATION is ON")
    endif()
    if(NOT EXISTS "${MAIW_QTVIEWERPRO_SOURCE_DIR}/CMakeLists.txt"
       OR NOT EXISTS "${MAIW_QTVIEWERPRO_SOURCE_DIR}/include/qtviewerpro/core/VolumeData.h"
       OR NOT EXISTS "${MAIW_QTVIEWERPRO_SOURCE_DIR}/include/qtviewerpro/io/MedicalVolumeLoaderRegistry.h")
        message(FATAL_ERROR "MAIW_QTVIEWERPRO_SOURCE_DIR must point to a plausible qt-viewer-pro source tree with CMakeLists.txt and public core/IO headers")
    endif()

    find_package(Qt6 REQUIRED COMPONENTS Core)

    add_subdirectory(
        "${MAIW_QTVIEWERPRO_SOURCE_DIR}"
        "${CMAKE_BINARY_DIR}/qt-viewer-pro-build"
        EXCLUDE_FROM_ALL
    )

    add_library(maiw-medical-image INTERFACE)
    target_link_libraries(maiw-medical-image
        INTERFACE
            qt-viewer-pro-core
            qt-viewer-pro-io
            Qt6::Core
    )

    if(BUILD_TESTING)
        set(MAIW_QTVIEWERPRO_UPSTREAM_TESTS
            test-volume-data
            test-volume-metadata
            test-anatomical-orientation
            test-volume-physical-coordinate-mapper
            test-volume-information
            test-volume-transfer-function
            test-slice-data
            test-slice-extractor
            test-mpr-coordinate-mapper
            test-mpr-orientation-label-mapper
            test-window-level-processor
            test-volume-resampler
            test-slice-image-converter
            test-raw-volume-loader
            test-nifti-volume-loader
            test-metaimage-volume-loader
            test-nrrd-volume-loader
        )
        set_tests_properties(${MAIW_QTVIEWERPRO_UPSTREAM_TESTS}
            DIRECTORY "${CMAKE_BINARY_DIR}/qt-viewer-pro-build"
            PROPERTIES DISABLED TRUE
        )
    endif()
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

if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
    add_library(maiw-cardiac-preprocessing
        src/cardiac/CardiacMriPreprocessing.cpp
    )

    target_include_directories(maiw-cardiac-preprocessing
        PUBLIC
            include
    )

    target_link_libraries(maiw-cardiac-preprocessing
        PUBLIC
            qt-viewer-pro-core
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(maiw-cardiac-preprocessing
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()
endif()

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
    if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
        add_executable(test-cardiac-mri-preprocessing
            tests/cardiac/TestCardiacMriPreprocessing.cpp
        )

        target_link_libraries(test-cardiac-mri-preprocessing
            PRIVATE
                maiw-cardiac-preprocessing
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-cardiac-mri-preprocessing
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME cardiac-mri-preprocessing-math
            COMMAND test-cardiac-mri-preprocessing
        )

        if(MAIW_CARDIAC_MRI_REAL_ED_PATH AND MAIW_CARDIAC_MRI_REAL_ES_PATH)
            add_executable(test-cardiac-mri-preprocessor-real-smoke
                tests/cardiac/TestCardiacMriPreprocessorRealSmoke.cpp
            )

            target_link_libraries(test-cardiac-mri-preprocessor-real-smoke
                PRIVATE
                    maiw-cardiac-preprocessing
                    qt-viewer-pro-io
                    Qt6::Core
            )

            target_compile_definitions(test-cardiac-mri-preprocessor-real-smoke
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                    MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-preprocessor-real-smoke
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME cardiac-mri-preprocessor-real-smoke
                COMMAND test-cardiac-mri-preprocessor-real-smoke
            )
        else()
            message(STATUS "MAIW_CARDIAC_MRI_REAL_ED_PATH/ES_PATH are not set; cardiac real preprocessing smoke test disabled")
        endif()

        # This exact preprocessing parity gate needs the golden package, real ACDC
        # training volumes, and the qt-viewer-pro IO integration in this parent block.
        if(MAIW_CARDIAC_MRI_PACKAGE_DIR AND MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR)
            find_package(OpenSSL REQUIRED COMPONENTS Crypto)

            add_executable(test-cardiac-mri-preprocessing-golden-parity
                tests/cardiac/TestCardiacMriPreprocessingGoldenParity.cpp
                tests/support/NpyReader.cpp
                tests/support/Sha256.cpp
            )

            target_include_directories(test-cardiac-mri-preprocessing-golden-parity
                PRIVATE
                    tests/support
            )

            target_compile_definitions(test-cardiac-mri-preprocessing-golden-parity
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR="${MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR}"
            )

            target_link_libraries(test-cardiac-mri-preprocessing-golden-parity
                PRIVATE
                    maiw-cardiac-preprocessing
                    qt-viewer-pro-io
                    Qt6::Core
                    nlohmann_json::nlohmann_json
                    OpenSSL::Crypto
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-preprocessing-golden-parity
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME cardiac-mri-preprocessing-golden-parity
                COMMAND test-cardiac-mri-preprocessing-golden-parity
            )

            # Phase 6 verifies the complete production path from real ED/ES
            # volumes through preprocessing and ONNX Runtime to frozen Python logits.
            add_executable(test-cardiac-mri-end-to-end-golden-parity
                tests/cardiac/TestCardiacMriEndToEndGoldenParity.cpp
                tests/support/NpyReader.cpp
                tests/support/Sha256.cpp
            )

            target_include_directories(test-cardiac-mri-end-to-end-golden-parity
                PRIVATE
                    tests/support
            )

            target_compile_definitions(test-cardiac-mri-end-to-end-golden-parity
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR="${MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR}"
            )

            target_link_libraries(test-cardiac-mri-end-to-end-golden-parity
                PRIVATE
                    maiw-cardiac-preprocessing
                    maiw-onnx-runtime
                    qt-viewer-pro-io
                    Qt6::Core
                    nlohmann_json::nlohmann_json
                    OpenSSL::Crypto
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-end-to-end-golden-parity
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-end-to-end-golden-parity
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-end-to-end-golden-parity
                COMMAND test-cardiac-mri-end-to-end-golden-parity
            )
        else()
            message(STATUS "MAIW_CARDIAC_MRI_PACKAGE_DIR or MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR is not set; real-data cardiac parity tests disabled")
        endif()
    endif()

    if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
        add_executable(test-qt-viewer-pro-dependency
            tests/integration/TestQtViewerProDependency.cpp
        )

        target_link_libraries(test-qt-viewer-pro-dependency
            PRIVATE
                maiw-medical-image
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-qt-viewer-pro-dependency
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME qt-viewer-pro-dependency-smoke
            COMMAND test-qt-viewer-pro-dependency
        )
    endif()

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
