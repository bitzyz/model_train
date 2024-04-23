# 设置项目名称和版本（可选）  
cmake_minimum_required(VERSION 3.17)  
project(MyProject VERSION 1.0)  
  
# 设置C++标准（例如C++17）  
set(CMAKE_CXX_STANDARD 17)  
set(CMAKE_CXX_STANDARD_REQUIRED True)  

find_package(OpenMP REQUIRED) 
  
# 添加包含目录  
include_directories(include)  
  
# 添加源文件目录  
file(GLOB_RECURSE SRC src/*.cc)
add_library (MyProject STATIC ${SRC})  
target_link_libraries(MyProject PRIVATE ${OpenMP_CXX_LIBRARIES})
  
# 添加可执行文件，并指定源文件  
add_executable(train main.cc)  

target_link_libraries(train MyProject ${OpenMP_CXX_LIBRARIES})
target_compile_options(train PRIVATE -Wno-pointer-arith)

# 设置OpenMP编译标志  
set_target_properties(train PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")  
set_target_properties(MyProject PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
