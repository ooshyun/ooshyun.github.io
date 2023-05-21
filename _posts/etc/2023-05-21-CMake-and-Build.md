---
title: Recipe CMake for Compile
key: 20230521
tags: etc
---

이번에 기업과제에서 코드 환경이 윈도우 Visual studio로 환경이었습니다. 저는 윈도우 환경이 없던 터라 빌드 시스템을 작게나마 만들어 보고자, 이번에는 `CMake를 이용한 빌드 시스템`에 대해서 정리하는 시간을 가져봤습니다. 더 자세히 알고 싶다면 레퍼런스를 참고하시길 바랍니다.

## 1. gcc vs make vs CMake

### 1.1 Compiler

프로그래밍 언어(High-level languange) 를 기계가 이해할 수 있는 언어(Low-level language, assembly, obeject code, machine code)로 번역해주는, Compiler 과정에서 가장 중요하는 역할을 하는 녀석이 바로 Compiler입니다. 필자가 많이 접하는 C/C++ 언어용 컴파일러로는 GCC C(gcc), GCC(g++), Clang, Clang++이 있습니다. 더 많은 컴파일을 알고 싶다면, [여기](https://en.wikipedia.org/wiki/List_of_compilers)로. arm64(arm 프로세서), x86_64(intel 프로세서)와 같은 OS 종류에서부터 최적화까지 컴파일러를 통해 할 수 있는데, 오늘은 **“어떻게 컴파일러를 이용하여 프로그래밍 언어를 빌드할 수 있는가?”** 에 대해서 정리해보고자 합니다.

빌드 과정으로 저는 CMake에 대해서 자세히 다룰 예정인데, 시작하기 전에 컴파일러 중에 gcc, make를 간단하게 짚고 넘어가 보겠습니다.

### 1.2 gcc

C, C++, Objectvie-C, Foran, Ada, Go, 그리고 D까지. libstdc++과 같은 라이브러리를 함꼐 사용하기 위해 GNU operating system을 위한 컴파일러로 GNU C 컴파일러 gcc를 많이 사용합니다. Mac, Linux, Window 모두 gcc/g++ 컴파일러는 있으니 한 번쯤 확인해보시면 필자의 경우 homebrew에 하나, /usr/bin/ 에 하나 있었습니다. 컴파일러 과정에 대해 을 알고 나면, 

```
Source Code → Preprocessing → Include header, Expand Macro(.i, .ii) 
→ Compiler(gcc, g++) → Assembly Code(.s) → Assembler(as) 
→ Machine Code(.o, .obj) → Linker(ld), Static Library(.lib, .a) 
→ Executable Machine Code(.exe)
```
그럼 어떻게 컴파일 해야 되는데? 의 질문 할 수 있는데, 그 해답이 바로 이 컴파일러 중 하나인 gcc를 사용하는 방법을 익히는 것 입니다. 오늘은 CMake를 다룰 예정이라 더 자세한 과정은 [이 링크](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html)를 참고하시면 되겠습니다.

### 1.3 Make

Make는 그럼 무엇일까요? 코드를 하나, 두개만 써놓은 것은 아니고 우리는 수십개의 파일을 자동적으로 컴파일 할 수 있도록 “군집화”해주는 utility 입니다. **makefile**이라는 파일을 통해 커맨드 창에 make만 치면 자동적으로 컴파일이 될 수 있게 해줍니다. 이 또한, 자세한 내용은 [이 링크](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html)를 참고하시길 바랍니다.

## 2. CMake

자자, 본격적으로 CMake에 대해서 이야기하도록 하겠습니다. 앞서 gcc와 Make를 먼저 언급한 것은 **CMake는 “generator of build systmes”**로 내용에 따라 make로 사용할 수 있는 makefile을 만들어주는 용도이기 때문입니다. 그럼 자연스레 make를 통해 빌드를 할 수 있고, make를 이용할 수 있다는 건 컴파일러를 이용할 수 있습니다. 그래서 필자의 경우는 gcc와 make가 아인 CMake를 통해서 Build System을 구축하고자 합니다.

```
.
├── Source1
|   ├── SubDir1-1
│   │   ├── Config.cmake
│   │   └── func11.cpp
|   ├── SubDir1-2
│   │   ├── Config.cmake
│   │   └── func12.cpp
    ...   ...
|		└── CMakeLists.txt

├── Source2
|   ├── SubDir2-1
│   │   ├── Config.cmake
│   │   └── func21.cpp
|   ├── SubDir2-2
│   │   ├── Config.cmake
│   │   └── func22.cpp
    ...   ...
|   ├── CmakeFunc1.cmake
|   ├── CmakeFunc2.cmake
|		└── CMakeLists.txt
└── CMakeLists.txt
```

### 2.2 CMake Basic Usage

#### #0 CMakelists.txt 는 프로젝트 명을 적어주기
    
가장 먼저 하는 것은 cmake 버전 조건과 프로젝트 명을 적는 것입니다(내가 누구인지는 알아야 파일도 실행을 하지... 빼먹지 말기!).

```
cmake_minimum_required(VERSION 3.0.0)
project(SourceAssignment VERSION 0.1.0)
```
    
#### #1 기본으로 변수명, 디버깅, 캐쉬에 대해 먼저 짚고 넘어가기

- 변수는? ${변수명}
- 메세지 출력(c에서 **printf**, c++에서 **cout**)은? `message(”내용”)`
- Cache처럼 저장/설정돼 있는 변수 확인은? `CMakeCache.txt`

예를 들어 보면, 설정한 `CMAKE_OSX_ARCHITECTURES` 를 확인하고 싶다면 아래처럼 사용할 수 있습니다.

```
# CMakelists.txt
message("CMAKE_OSX_ARCHITECTURES: " ${CMAKE_OSX_ARCHITECTURES})
```

#### #2 음, 그럼 변수를 설정할 수도 있어?

스크립트에서 하는 방법, 터미널에서 하는 방법 그리고`CMakeCache.txt`에 저장하는 방법(이건 vscode 이용시 CMakeCache.txt에 저장해서 빌드할 때 이용가능) 이 있습니다.

- CMakeLists.txt 에서 설정: `set(VAR_NAME VAR)`
- 터미널에서 설정하기:`cmake CMakeLists.txt -DVAR_NAME:VAR_DTYPE=VAR`
- CMakeCache.txt에 저장하기: `set(VAR_NAME "" CACHE STRING "Description of the argument")`

예를 들어 보겠습니다.

- cmake를 실행할때 변수 ARGU_BUILD_TYPE 변수명으로 “exe”를 받고 싶다면,
    
    ```bash
    cmake CMakeLists.txt -DARGU_BUILD_TYPE:STRING=exe
    ```
    
- 스크립트에서 변수 ARGU_BUILD_TYPE 를 `CMakeCache.txt`에 저장하고 싶다면,
    
    ```
    # CMakelists.txt
    set(ARGU_BUILD_TYPE "lib" CACHE STRING "Description of the argument")
    message("Building for " ${ARGU_BUILD_TYPE})
    ```
        

#### #3 그럼 If문도 가능한가?

당연히 가능합니다. If문 조건으로 많이 사용하는 bool같은 녀석인 **option**과 If문 사용법은 다음과 같습니다.

- option: `option(LIBRARY "Build library" ON)`
- If, else if, else, endif
    
    ```
    # CMakelists.txt
    if(<condition>)
        <commands>
    elseif(<condition>) # optional block, can be repeated
        <commands>
    else()              # optional block
        <commands>
    endif()
    ```
    
예시를 들어 보면 아래는 LIBRARY를 ON해서 “Build libarary…” 메세지를 출력하는 스크립트입니다.

```
# CMakelists.txt
# 만약 ON을 하지 않으면? OFF일 것이다. 

option(LIBRARY "Build library" ON)

if(LIBRARY)
    message(STATUS "Build Library...")
endif()
```
    
여기까지 하면 간단한 스크립트를 짤 수 있을 것입니다. 그러면 이제 작게 빌드시스템을 만들어 보겠습니다.

### 2.3 CMake Build System

필자의 경우는 코드에서 우선 라이브러리로 빌드해, 그 라이브러리를 링크한 실행파일을 만드는 게 목적이었습니다. 독립적으로 빌드를 하는 경우와 전체를 빌드하는 경우, 두 가지 경우를 만들기 위해 아래와 같이 질문이 나왔고, 각각의 답을 통해 전체 빌드 시스템을 짜보도록 하겠습니다.

#### #1 빌드 시스템을 통해 쉽게 “모드”를 설정할 수 없을까?

Bash 스크립트를 통해 Argument를 받아 library, execute, library + execute 모드로 나눌 수 있습니다.
    
#### #2 빌드 시스템을 통해 여러개의 라이브러리와 실행파일을 만들 수 있을까?

CMake의 경우, 여러개의 라이브러리와 실행파일을 빌드가 가능합니다. 이는 아래 두 개를 이용하면 됩니다.

- add_library(LIBRARY_NAME LIBRARY_TYPE LIBRARY_TARGET_FILES)
- add_executable(EXEUTE_NAME EXCUTE_TYPE EXCUTE_TARGET_FILES)

그리고 실행파일의 경우 라이브러리와 링크해야 하므로 아래의 경우를 사용하면 됩니다.

- 실행파일을 라이브러에 링크해 빌드하는 경우
    
    ```
    set(VAR_LIBRARY_PATH "LIBRARY_PATH")
    add_executable(EXEUTE_NAME EXCUTE_TYPE EXCUTE_TARGET_FILES)
    target_link_libraries(EXEUTE_NAME ${VAR_LIBRARY_PATH})
    ```
        
#### #3 빌드 시스템을 통해 OS아키텍처에 따른 실행파일을 만들 수 있을까?

필자의 경우 x86_64와 arm64가 있었는데 이는 CMAKE_OSX_ARCHITECTURES를 설정하면 됩니다.

```
# CMakelists.txt
set(CMAKE_OSX_ARCHITECTURES arm64)
```
    
x86_64와 arm64, 두 가지 환경을 위한 컴파일의 경우 아래와 같이 설정합니다. 

```
# CMakelists.txt
set(CMAKE_OSX_ARCHITECTURES x86_64;arm64)
```

#### #4 빌드 시스템을 통해 Argument(ex. ls -al 에서 -al과 같은 요소)를 받을 수 있을까?

네 받을 수 있습니다. DARGU_BUILD_TYPE 를 통해 가능한데, 이전에 설명한 바와 같이 ```bash cmake CMakeLists.txt -DVAR_NAME:VAR_DTYPE=VAR``` 를 이용해 가능합니다.

```bash
cmake -DARGU_BUILD_TYPE:STRING=lib
```
    
#### #5 빌드 시스템을 각 폴더 별로 독립적으로 구분지어서 작성할 수 없을까?

이 경우 세 가지가 경우가 있습니다. 첫 번째는 빌드 시스템의 역할을 하는 경우, 두 번째는 Configuration의 역할을 하는 경우, 그리고 마지막은 function을 하는 경우입니다. **빌드 시스템의 경우 CMakeLists.txt, Configuration의 경우 Config.cmake를 이용하면 됩니다. (function의 경우 [여기](https://github.com/ARM-software/CMSIS-DSP)를 참고!)

예를 들어 필자의 상황은 아래와 같다고 가정해보겠습니다.

```
.
├── Source
|   ├── SubDir
│   │   ├── Config.cmake
│   │   └── main.cpp
|		└── CMakeLists.txt
└── CMakeLists.txt
```

이 경우, 첫 번째는 빌드 시스템의 역할을 하는 경우는 `add_subdirectory`를 이용해서 아래와 같이 사용하면 됩니다. (`${PROJECT_SOURCE_DIR}` 의 경우는 현재 프로젝트 경로를 의미하니 Message로 확인해보자. 자세한 내용은 [여기](https://cmake.org/cmake/help/v2.8.8/cmake.html#section_VariablesThatDescribetheSystem)에서!)

```
// ./CMakeLists.txt
add_subdirectory(${PROJECT_SOURCE_DIR}/Source)
```

두 번째, Configuration의 역할을 하는 경우는 `include` 를 이용해서 아래와 같이 사용하면 된다.

```
// ./Source/CMakeLists.txt
include(${PROJECT_SOURCE_DIR}/Source)
```

- 이렇게 Configuration을 이용하여 타겟 소스 파일은 서브폴더에서도 지정이 가능하다.
    
    ```
    // ./Source/CMakeLists.txt
    add_executable(EXECUTE_NAME)
    include(EXECUTE_NAME/Config.cmake)
    ```
    
    ```
    // ./Source/SubDir/Config.cmake
    target_sources(EXECUTE_NAME "${PROJECT_SOURCE_DIR}/SubDir/main.cpp")
    ```
        
#### #6 빌드 시스템이 컴파일러와 버전을 지정할 수 있을까?

```
# Set Compiler
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```
    
    
사실, CMake의 경우 처음에 소개한 두 링크만 천천히 읽어도 따라가는데 크게 지장이 없었습니다. 그리고 세세한 옵션의 경우 그 때 그때 찾던지, Chatgpt를 이용하면 될 듯 싶습니다. CMake를 이용하면 이미 있는 라이브러리나 프레임워크를 찾는 `find_package`, `크로스 컴파일을 위한 CMake` 와 같은 것 또한 가능할 것인데, 위에서 사용한 예시들은 [이 링크](https://github.com/ooshyun/Make-and-CMake-Examples/tree/master/examples/1))에 전체 코드를 첨부해놨습니다. 워낙 레페런스의 글들이 잘 정리해놓으셨다보니 이 글의 목적은 CMake에 대한 전반적인 그림을 그리는 것과 정리를 목적으로 이야기 해봤습니다. 다른 기능들을 해야할 날이 온다면 추가하는 걸로하고 여기서 글을 마치겠습니다.

## 3. Reference
- [Make-and-CMake-Examples/examples/1 at master · ooshyun/Make-and-CMake-Examples](https://github.com/ooshyun/Make-and-CMake-Examples/tree/master/examples/1)
- [https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html](https://www3.ntu.edu.sg/home/ehchua/programming/cpp/gcc_make.html)
- [https://gist.github.com/luncliff/6e2d4eb7ca29a0afd5b592f72b80cb5c](https://gist.github.com/luncliff/6e2d4eb7ca29a0afd5b592f72b80cb5c)