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
include(GNUInstallDirs)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root directory of an ONNX Runtime C++ distribution")
set(MAIW_CARDIAC_MRI_PACKAGE_DIR "" CACHE PATH "External cardiac MRI deployment package root")
# Real ACDC data remains external so portable CI can configure without private medical datasets.
set(MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR "" CACHE PATH "External real ACDC training dataset root")
set(MAIW_CARDIAC_MRI_REAL_ED_PATH "" CACHE FILEPATH "Optional real standalone ED volume for cardiac preprocessing smoke testing")
set(MAIW_CARDIAC_MRI_REAL_ES_PATH "" CACHE FILEPATH "Optional real standalone ES volume for cardiac preprocessing smoke testing")
set(MAIW_QTVIEWERPRO_SOURCE_DIR "" CACHE PATH "Local qt-viewer-pro source directory")
option(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION "Enable local qt-viewer-pro core/IO source integration" OFF)

if(MAIW_CARDIAC_MRI_PACKAGE_DIR)
    foreach(MAIW_CARDIAC_MRI_RUNTIME_ASSET
            classifier.onnx
            deployment.json
            class_mapping.json)
        if(NOT EXISTS "${MAIW_CARDIAC_MRI_PACKAGE_DIR}/${MAIW_CARDIAC_MRI_RUNTIME_ASSET}")
            message(FATAL_ERROR "Required cardiac MRI runtime asset is missing: ${MAIW_CARDIAC_MRI_PACKAGE_DIR}/${MAIW_CARDIAC_MRI_RUNTIME_ASSET}")
        endif()
    endforeach()
endif()

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

    find_package(Qt6 REQUIRED COMPONENTS Core Concurrent Widgets)

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

    add_library(maiw-viewer-widgets
        include/maiw/viewer/MprViewerWidget.h
        include/maiw/viewer/SliceViewerWidget.h
        include/maiw/viewer/VolumeLoadWorkflow.h
        include/maiw/viewer/VolumeRenderingWidget.h
        include/maiw/viewer/ViewerWorkspaceWidget.h
        src/viewer/MprViewerWidget.cpp
        src/viewer/SliceViewerWidget.cpp
        src/viewer/VolumeLoadWorkflow.cpp
        src/viewer/VolumeRenderingWidget.cpp
        src/viewer/ViewerWorkspaceWidget.cpp
    )

    set_target_properties(maiw-viewer-widgets
        PROPERTIES
            AUTOMOC ON
    )

    target_include_directories(maiw-viewer-widgets
        PUBLIC
            include
    )

    target_link_libraries(maiw-viewer-widgets
        PUBLIC
            qt-viewer-pro-core
            Qt6::Core
            Qt6::Widgets
        PRIVATE
            qt-viewer-pro-io
            qt-viewer-pro-processing
            qt-viewer-pro-render
            Qt6::Concurrent
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(maiw-viewer-widgets
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

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

add_library(maiw-cardiac-classification
    src/cardiac/CardiacMriDeploymentMetadata.cpp
    src/cardiac/CardiacMriClassificationResult.cpp
)

target_include_directories(maiw-cardiac-classification
    PUBLIC
        include
)

target_link_libraries(maiw-cardiac-classification
    PRIVATE
        nlohmann_json::nlohmann_json
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(maiw-cardiac-classification
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

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

if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
    target_sources(maiw-cardiac-classification
        PRIVATE
            src/cardiac/CardiacMriClassificationService.cpp
    )

    target_link_libraries(maiw-cardiac-classification
        PUBLIC
            maiw-cardiac-preprocessing
            maiw-onnx-runtime
    )

    add_library(maiw-cardiac-classification-workflow
        include/maiw/qt/CardiacMriClassificationResultWidget.h
        include/maiw/qt/CardiacMriClassificationWindow.h
        include/maiw/qt/CardiacMriClassificationWorkflow.h
        src/qt/CardiacMriClassificationResultWidget.cpp
        src/qt/CardiacMriClassificationWindow.cpp
        src/qt/CardiacMriClassificationWorkflow.cpp
    )

    set_target_properties(maiw-cardiac-classification-workflow
        PROPERTIES
            AUTOMOC ON
    )

    target_include_directories(maiw-cardiac-classification-workflow
        PUBLIC
            include
    )

    target_link_libraries(maiw-cardiac-classification-workflow
        PUBLIC
            maiw-cardiac-classification
            Qt6::Core
            Qt6::Widgets
        PRIVATE
            qt-viewer-pro-io
            Qt6::Concurrent
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(maiw-cardiac-classification-workflow
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

    add_library(maiw-workstation-ui
        include/maiw/qt/MedicalAiWorkstationWindow.h
        src/qt/MedicalAiWorkstationWindow.cpp
    )

    target_include_directories(maiw-workstation-ui
        PUBLIC
            include
    )

    target_link_libraries(maiw-workstation-ui
        PUBLIC
            maiw-cardiac-classification-workflow
            maiw-viewer-widgets
            Qt6::Widgets
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(maiw-workstation-ui
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

    add_executable(medical-ai-workstation
        src/app/main.cpp
    )

    target_link_libraries(medical-ai-workstation
        PRIVATE
            maiw-workstation-ui
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(medical-ai-workstation
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

    file(RELATIVE_PATH MAIW_INSTALL_LIBDIR_FROM_BINDIR
        "${CMAKE_INSTALL_FULL_BINDIR}"
        "${CMAKE_INSTALL_FULL_LIBDIR}"
    )
    file(RELATIVE_PATH MAIW_INSTALL_DATADIR_FROM_BINDIR
        "${CMAKE_INSTALL_FULL_BINDIR}"
        "${CMAKE_INSTALL_FULL_DATADIR}"
    )

    target_compile_definitions(medical-ai-workstation
        PRIVATE
            MAIW_INSTALL_DATADIR_FROM_BINDIR="${MAIW_INSTALL_DATADIR_FROM_BINDIR}"
    )

    set_target_properties(medical-ai-workstation
        PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            INSTALL_RPATH "\$ORIGIN/${MAIW_INSTALL_LIBDIR_FROM_BINDIR}"
            INSTALL_RPATH_USE_LINK_PATH FALSE
    )

    install(TARGETS medical-ai-workstation
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    if(Qt6Core_VERSION VERSION_GREATER_EQUAL 6.5)
        qt_generate_deploy_app_script(
            TARGET medical-ai-workstation
            OUTPUT_SCRIPT MAIW_QT_DEPLOY_SCRIPT
            NO_TRANSLATIONS
        )
        install(SCRIPT "${MAIW_QT_DEPLOY_SCRIPT}")
    else()
        message(STATUS "Private Qt runtime deployment requires Qt 6.5 or newer; skipping deployment for Qt ${Qt6Core_VERSION}")
    endif()

    install(CODE "
        file(INSTALL \"${ONNXRUNTIME_LIBRARY}\"
            DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}\"
            FOLLOW_SYMLINK_CHAIN
        )
    ")

    if(MAIW_CARDIAC_MRI_PACKAGE_DIR)
        install(FILES
            "${MAIW_CARDIAC_MRI_PACKAGE_DIR}/classifier.onnx"
            "${MAIW_CARDIAC_MRI_PACKAGE_DIR}/deployment.json"
            "${MAIW_CARDIAC_MRI_PACKAGE_DIR}/class_mapping.json"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/medical-ai-workstation/cardiac_mri_pathology"
        )
    endif()
endif()

if(BUILD_TESTING)
    add_executable(test-cardiac-mri-classification
        tests/cardiac/TestCardiacMriClassification.cpp
    )

    target_link_libraries(test-cardiac-mri-classification
        PRIVATE
            maiw-cardiac-classification
    )

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test-cardiac-mri-classification
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
        )
    endif()

    add_test(
        NAME cardiac-mri-classification-contract
        COMMAND test-cardiac-mri-classification
    )

    if(MAIW_ENABLE_QTVIEWERPRO_INTEGRATION)
        add_executable(test-slice-viewer-widget
            tests/viewer/TestSliceViewerWidget.cpp
        )

        target_link_libraries(test-slice-viewer-widget
            PRIVATE
                maiw-viewer-widgets
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-slice-viewer-widget
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME slice-viewer-widget
            COMMAND test-slice-viewer-widget
        )

        set_tests_properties(slice-viewer-widget
            PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        add_executable(test-mpr-viewer-widget
            tests/viewer/TestMprViewerWidget.cpp
        )

        target_link_libraries(test-mpr-viewer-widget
            PRIVATE
                maiw-viewer-widgets
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-mpr-viewer-widget
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME mpr-viewer-widget
            COMMAND test-mpr-viewer-widget
        )

        set_tests_properties(mpr-viewer-widget
            PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        add_executable(test-volume-load-workflow
            tests/viewer/TestVolumeLoadWorkflow.cpp
        )

        target_link_libraries(test-volume-load-workflow
            PRIVATE
                maiw-viewer-widgets
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-volume-load-workflow
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME volume-load-workflow
            COMMAND test-volume-load-workflow
        )

        add_executable(test-volume-rendering-widget
            tests/viewer/TestVolumeRenderingWidget.cpp
        )

        target_link_libraries(test-volume-rendering-widget
            PRIVATE
                maiw-viewer-widgets
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-volume-rendering-widget
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME volume-rendering-widget
            COMMAND test-volume-rendering-widget
        )

        set_tests_properties(volume-rendering-widget
            PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        add_executable(test-viewer-workspace-widget
            tests/viewer/TestViewerWorkspaceWidget.cpp
        )

        target_link_libraries(test-viewer-workspace-widget
            PRIVATE
                maiw-viewer-widgets
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-viewer-workspace-widget
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME viewer-workspace-widget
            COMMAND test-viewer-workspace-widget
        )

        set_tests_properties(viewer-workspace-widget
            PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        if(MAIW_CARDIAC_MRI_REAL_ED_PATH
           AND EXISTS "${MAIW_CARDIAC_MRI_REAL_ED_PATH}")
            add_executable(test-slice-viewer-widget-real-volume
                tests/viewer/TestSliceViewerWidgetRealVolume.cpp
            )

            target_compile_definitions(test-slice-viewer-widget-real-volume
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
            )

            target_link_libraries(test-slice-viewer-widget-real-volume
                PRIVATE
                    maiw-viewer-widgets
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-slice-viewer-widget-real-volume
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME slice-viewer-widget-real-volume
                COMMAND test-slice-viewer-widget-real-volume
            )

            set_tests_properties(slice-viewer-widget-real-volume
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            add_executable(test-mpr-viewer-widget-real-volume
                tests/viewer/TestMprViewerWidgetRealVolume.cpp
            )

            target_compile_definitions(test-mpr-viewer-widget-real-volume
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
            )

            target_link_libraries(test-mpr-viewer-widget-real-volume
                PRIVATE
                    maiw-viewer-widgets
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-mpr-viewer-widget-real-volume
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME mpr-viewer-widget-real-volume
                COMMAND test-mpr-viewer-widget-real-volume
            )

            set_tests_properties(mpr-viewer-widget-real-volume
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            add_executable(test-volume-rendering-widget-real-volume
                tests/viewer/TestVolumeRenderingWidgetRealVolume.cpp
            )

            target_compile_definitions(test-volume-rendering-widget-real-volume
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
            )

            target_link_libraries(test-volume-rendering-widget-real-volume
                PRIVATE
                    maiw-viewer-widgets
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-volume-rendering-widget-real-volume
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME volume-rendering-widget-real-volume
                COMMAND test-volume-rendering-widget-real-volume
            )

            set_tests_properties(volume-rendering-widget-real-volume
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            add_executable(test-viewer-workspace-widget-real-volume
                tests/viewer/TestViewerWorkspaceWidgetRealVolume.cpp
            )

            target_compile_definitions(test-viewer-workspace-widget-real-volume
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
            )

            target_link_libraries(test-viewer-workspace-widget-real-volume
                PRIVATE
                    maiw-viewer-widgets
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-viewer-workspace-widget-real-volume
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            add_test(
                NAME viewer-workspace-widget-real-volume
                COMMAND test-viewer-workspace-widget-real-volume
            )

            set_tests_properties(viewer-workspace-widget-real-volume
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            add_executable(test-volume-rendering-widget-gpu
                tests/viewer/TestVolumeRenderingWidgetGpu.cpp
            )

            target_compile_definitions(test-volume-rendering-widget-gpu
                PRIVATE
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
            )

            target_link_libraries(test-volume-rendering-widget-gpu
                PRIVATE
                    maiw-viewer-widgets
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-volume-rendering-widget-gpu
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()
        else()
            message(STATUS "MAIW_CARDIAC_MRI_REAL_ED_PATH is not set or does not exist; real-volume viewer tests disabled")
        endif()

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

        add_executable(test-cardiac-mri-classification-result-widget
            tests/qt/TestCardiacMriClassificationResultWidget.cpp
        )

        target_link_libraries(test-cardiac-mri-classification-result-widget
            PRIVATE
                maiw-cardiac-classification-workflow
        )

        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test-cardiac-mri-classification-result-widget
                PRIVATE
                    -Wall
                    -Wextra
                    -Wpedantic
            )
        endif()

        add_test(
            NAME cardiac-mri-classification-result-widget
            COMMAND test-cardiac-mri-classification-result-widget
        )

        set_tests_properties(cardiac-mri-classification-result-widget
            PROPERTIES
                ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
        )

        if(MAIW_CARDIAC_MRI_PACKAGE_DIR)
            add_executable(test-cardiac-mri-classification-window
                tests/qt/TestCardiacMriClassificationWindow.cpp
            )

            target_compile_definitions(test-cardiac-mri-classification-window
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
            )

            if(MAIW_CARDIAC_MRI_REAL_ED_PATH AND MAIW_CARDIAC_MRI_REAL_ES_PATH)
                target_compile_definitions(test-cardiac-mri-classification-window
                    PRIVATE
                        MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                        MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
                )
            endif()

            target_link_libraries(test-cardiac-mri-classification-window
                PRIVATE
                    maiw-cardiac-classification-workflow
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-classification-window
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-classification-window
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-classification-window
                COMMAND test-cardiac-mri-classification-window
            )

            set_tests_properties(cardiac-mri-classification-window
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            add_executable(test-medical-ai-workstation-window
                tests/qt/TestMedicalAiWorkstationWindow.cpp
            )

            target_compile_definitions(test-medical-ai-workstation-window
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
            )

            target_link_libraries(test-medical-ai-workstation-window
                PRIVATE
                    maiw-workstation-ui
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-medical-ai-workstation-window
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-medical-ai-workstation-window
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME medical-ai-workstation-window
                COMMAND test-medical-ai-workstation-window
            )

            set_tests_properties(medical-ai-workstation-window
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
            )

            if(MAIW_CARDIAC_MRI_REAL_ED_PATH
               AND EXISTS "${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
               AND MAIW_CARDIAC_MRI_REAL_ES_PATH
               AND EXISTS "${MAIW_CARDIAC_MRI_REAL_ES_PATH}")
                add_executable(test-medical-ai-workstation-study-coordination
                    tests/qt/TestMedicalAiWorkstationStudyCoordination.cpp
                )

                target_compile_definitions(test-medical-ai-workstation-study-coordination
                    PRIVATE
                        MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                        MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                        MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
                )

                target_link_libraries(test-medical-ai-workstation-study-coordination
                    PRIVATE
                        maiw-workstation-ui
                )

                if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                    target_compile_options(test-medical-ai-workstation-study-coordination
                        PRIVATE
                            -Wall
                            -Wextra
                            -Wpedantic
                    )
                endif()

                set_target_properties(test-medical-ai-workstation-study-coordination
                    PROPERTIES
                        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
                )

                add_test(
                    NAME medical-ai-workstation-study-coordination
                    COMMAND test-medical-ai-workstation-study-coordination
                )

                set_tests_properties(medical-ai-workstation-study-coordination
                    PROPERTIES
                        ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
                )
            else()
                message(STATUS "Real ED/ES paths are not both configured and available; Medical AI Workstation study coordination test disabled")
            endif()
        else()
            message(STATUS "MAIW_CARDIAC_MRI_PACKAGE_DIR is not set; cardiac classification window test disabled")
        endif()

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

        if(MAIW_CARDIAC_MRI_PACKAGE_DIR
           AND MAIW_CARDIAC_MRI_REAL_ED_PATH
           AND MAIW_CARDIAC_MRI_REAL_ES_PATH)
            add_executable(test-cardiac-mri-classification-service
                tests/cardiac/TestCardiacMriClassificationService.cpp
            )

            target_compile_definitions(test-cardiac-mri-classification-service
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                    MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
            )

            target_link_libraries(test-cardiac-mri-classification-service
                PRIVATE
                    maiw-cardiac-classification
                    qt-viewer-pro-io
                    Qt6::Core
                    nlohmann_json::nlohmann_json
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-classification-service
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-classification-service
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-classification-service-smoke
                COMMAND test-cardiac-mri-classification-service
            )

            add_executable(test-cardiac-mri-classification-workflow
                tests/qt/TestCardiacMriClassificationWorkflow.cpp
            )

            target_compile_definitions(test-cardiac-mri-classification-workflow
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                    MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
            )

            target_link_libraries(test-cardiac-mri-classification-workflow
                PRIVATE
                    maiw-cardiac-classification-workflow
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-classification-workflow
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-classification-workflow
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-classification-workflow
                COMMAND test-cardiac-mri-classification-workflow
            )

            add_executable(test-cardiac-mri-classification-shutdown
                tests/qt/TestCardiacMriClassificationShutdown.cpp
            )

            target_compile_definitions(test-cardiac-mri-classification-shutdown
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_REAL_ED_PATH="${MAIW_CARDIAC_MRI_REAL_ED_PATH}"
                    MAIW_CARDIAC_MRI_REAL_ES_PATH="${MAIW_CARDIAC_MRI_REAL_ES_PATH}"
            )

            target_link_libraries(test-cardiac-mri-classification-shutdown
                PRIVATE
                    maiw-cardiac-classification-workflow
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-classification-shutdown
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-classification-shutdown
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-classification-shutdown
                COMMAND test-cardiac-mri-classification-shutdown
            )

            set_tests_properties(cardiac-mri-classification-shutdown
                PROPERTIES
                    ENVIRONMENT "QT_QPA_PLATFORM=offscreen"
                    TIMEOUT 30
            )
        else()
            message(STATUS "Cardiac package or real ED/ES paths are not set; cardiac classification service/workflow tests disabled")
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

            # Phase 7 validates the complete public cardiac classification service
            # against the same seven frozen Python inference references.
            add_executable(test-cardiac-mri-classification-service-golden-parity
                tests/cardiac/TestCardiacMriClassificationServiceGoldenParity.cpp
                tests/support/NpyReader.cpp
                tests/support/Sha256.cpp
            )

            target_include_directories(test-cardiac-mri-classification-service-golden-parity
                PRIVATE
                    tests/support
            )

            target_compile_definitions(test-cardiac-mri-classification-service-golden-parity
                PRIVATE
                    MAIW_CARDIAC_MRI_PACKAGE_DIR="${MAIW_CARDIAC_MRI_PACKAGE_DIR}"
                    MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR="${MAIW_CARDIAC_MRI_ACDC_TRAINING_DIR}"
            )

            target_link_libraries(test-cardiac-mri-classification-service-golden-parity
                PRIVATE
                    maiw-cardiac-classification
                    qt-viewer-pro-io
                    Qt6::Core
                    nlohmann_json::nlohmann_json
                    OpenSSL::Crypto
            )

            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(test-cardiac-mri-classification-service-golden-parity
                    PRIVATE
                        -Wall
                        -Wextra
                        -Wpedantic
                )
            endif()

            set_target_properties(test-cardiac-mri-classification-service-golden-parity
                PROPERTIES
                    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
            )

            add_test(
                NAME cardiac-mri-classification-service-golden-parity
                COMMAND test-cardiac-mri-classification-service-golden-parity
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
