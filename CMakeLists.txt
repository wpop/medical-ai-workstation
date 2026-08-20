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
