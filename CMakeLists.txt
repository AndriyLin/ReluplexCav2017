cmake_minimum_required(VERSION 3.10)
project(Reluplex)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-MMD -Wall -Wextra -Werror -Wno-deprecated")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-const-variable")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")  # not recognized by clang.. on mac
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

include_directories(common)
include_directories(nnet)
include_directories(reluplex)
include_directories(glpk-4.60/installed/include)
link_directories(glpk-4.60/installed/lib)

add_library(nnet nnet/nnet.cpp)


add_executable(RunReluplex reluplex/main.cpp)
target_link_libraries(RunReluplex nnet glpk)

add_executable(prop1 check_properties/property1/main.cpp)
target_link_libraries(prop1 nnet glpk)

add_executable(prop2 check_properties/property2/main.cpp)
target_link_libraries(prop2 nnet glpk)

add_executable(prop3 check_properties/property3/main.cpp)
target_link_libraries(prop3 nnet glpk)

add_executable(prop4 check_properties/property4/main.cpp)
target_link_libraries(prop4 nnet glpk)

add_executable(prop5 check_properties/property5/main.cpp)
target_link_libraries(prop5 nnet glpk)

add_executable(prop6a check_properties/property6a/main.cpp)
target_link_libraries(prop6a nnet glpk)

add_executable(prop6b check_properties/property6b/main.cpp)
target_link_libraries(prop6b nnet glpk)

add_executable(prop7 check_properties/property7/main.cpp)
target_link_libraries(prop7 nnet glpk)

add_executable(prop8 check_properties/property8/main.cpp)
target_link_libraries(prop8 nnet glpk)

add_executable(prop9 check_properties/property9/main.cpp)
target_link_libraries(prop9 nnet glpk)

add_executable(prop10 check_properties/property10/main.cpp)
target_link_libraries(prop10 nnet glpk)

add_executable(adversarial check_properties/adversarial/main.cpp)
target_link_libraries(adversarial nnet glpk)

add_executable(birth_decade check_properties/xlexp/birth_decade.cpp)
target_link_libraries(birth_decade nnet glpk)

add_executable(birth_decade_top3 check_properties/xlexp/birth_decade_top3.cpp)
target_link_libraries(birth_decade_top3 nnet glpk)

add_executable(birth_robust check_properties/xlexp/birth_robust.cpp)
target_link_libraries(birth_robust nnet glpk)

add_executable(birth_local_robust check_properties/xlexp/birth_local_robust.cpp)
target_link_libraries(birth_local_robust nnet glpk)

add_executable(nlp_exp check_properties/xlexp/nlp_exp.cpp)
target_link_libraries(nlp_exp nnet glpk)

add_executable(paper_prop check_properties/xlexp/paper_prop.cpp)
target_link_libraries(paper_prop nnet glpk)

add_executable(paper_robust check_properties/xlexp/paper_robust.cpp)
target_link_libraries(paper_robust nnet glpk)

add_executable(sentiment_prop check_properties/xlexp/sentiment_prop.cpp)
target_link_libraries(sentiment_prop nnet glpk)
