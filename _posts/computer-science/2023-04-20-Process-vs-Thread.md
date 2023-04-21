---
title: Process and Thread
aside:
    toc: true
key: 20230420
tags: CS OS
---
**프로세스 vs 쓰레드 vs 코어**, 오늘 OS에서 CPU나 I/O로 가서 명령을 처리하는 프로세스와 스레드, 그리고 이들을 처리해주는 코어에 대해서 이야기해보려고 합니다. 특히 빠른 시간내에 많은 양의 작업을 처리하기 위해서는 멀티프로세스와 멀티스레드를 사용하는데요, 이 녀석들은 작업관리자에서 열었을 때 프로세스와 스레드는 많이 볼 수 있죠. 코어는 컴퓨터를 구매하는 경우에 8코어 10코어와 같이 많이 볼 수 있구요(M1 Mac살 때 고민했었던 게 생각나네요). 그런 프로세스랑 스레드, 그리고 코어는 어떻게 다른 걸까요? 차근차근 알아보도록 하겠습니다. 

<!--more-->

- 본 내용은 [쉬운코드 유튜브 영상](https://www.youtube.com/@ez.)에 내용의 흐름을 따라갔습니다. 레퍼런스에 각 부분마다 링크가 달려있으니 참고 하시길 바랍니다.

- 큰 흐름은 **“Process -> Multi-programming → Multi-tasking → Thread → Multi-processing/Multi-threading”** 입니다.

## 1. 프로세스

우리는 어플리케이션을 만들죠. 여기서 코드로 짜여진 명령어는 처리되기 위해 프로세스에 이것저것을 담겨서 하드웨어로 보내집니다. 그리고 보내지기 위해 공간을 할당 받아야 겠죠? 이 할당받는 **독립적인** 메모리 공간을 **PCB(Process Control Block)**라고 합니다. 아래와 같이요.

<p>
    <img src="/assets/images/post/cs/process-thread/PCB.png" width="200" height="500" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"> ChatGPT가 그려준 PCB(Process Control Block)</em>
    </p>
</p>   

각 PCB는 프로세스의 주소, 상태, ID, 다음 Instruction의 주소, 레지스터 등등을 가지고 있죠.  이 놈들이 바로 우리가 하는 코드(명령, Instruction)을 커널을 통해 CPU코어에 전달돼 실행되는 명령어입니다.

그럼 우리는 지금 명령어들을 코어에서 처리하기 위해서 이 과정을 하고 있죠? 그런데, 이 프로세스가 만약 메모리를 읽으러 가버리면 그 동안에 코어는 비게 되게 됩니다. 그럼 남는 시간에 코어는 놀고 있게되니, 이건 리소스 낭비일 수 밖에 없죠. 그렇게 하나만 말고 “**여러 개를 사용해보자”** 해서, 여러개의 프로세스를 사용하게 됩니다. 이를 **멀티 프로그래밍, Multi-Programming**이라고 부릅니다. 

<p>
    <img src="/assets/images/post/cs/process-thread/multi-programming.png" width="400" height="200" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

하지만 또 여기서 앞선 프로세스의 시간이 길어지게 된다면, 그 다음 작업 B는 A가 끝날 때까지 계속 기다려야 하는 문제가 있습니다. 그래서 프로세스가 CPU를 점유하는 시간이 길어지는 것을 해결하고자 한 시스템이 **멀티태스킹(Multi-tasking)**입니다. **프로세스가 CPU에서 머무는 시간을 매우 짧은 시간(Quantom)**으로 가져가자고 한 것이죠. 이는 각 프로세스의 **응답시간을 최소화시키는 데 목적**이 있습니다. 

<p>
    <img src="/assets/images/post/cs/process-thread/multi-tasking.png" width="400" height="300" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

### 1.1 **동시성(Concurrency)와 병렬성(Parallelism)**

**동시성**(**Concurrency**)에 대해서 먼저 언급하고 지나갈까 합니다. 동시성하면 먼저 떠오르는 개념이 “병렬처리(**병렬성** **Parallelism** 이라고 부릅니다)”일 수 있겠는데요, 정확하게 동시성은 병렬처리보다는 아주 짧은 시간에 여러가지 일을 동시에 처리하는 것을 의미하는 겁니다. 

그림에서 A와 B의 일을 처리하기 위해서는 다음과 같이 일이 처리가 될 수 있겠죠? 그럼 이 때 각 테스트가 처리되는 시간은 얼마나 걸릴까요? Intel Core i9-13900K CPU기준으로 이 “코어”의 클럭은 1초에 3GHz, 1초에 30억개의 연산을 보낼 수 있습니다(정확하게 모든 연산 하나하나가 이 만큼 속도라고 말하기는 어렵지만 얼추 가늠해볼게요). 그러면 1개 연산에는 0.3 ns가 되는데, 어마어마하죠? 이 짧은 시간을 이용해 동시성이란 이처럼 **“아주 짧은 전환으로 여러가지 일을 동시에 처리하는 것처럼 보이는 것”**을 말합니다(병렬 처리는 단어 그대로 병.렬. 입니다). 

하지만 더 많은 작업을 처리하기 위해, 또 여기서 아쉬운 점 몇가지가 남습니다.

- 하나의 프로세스에서도 동시에 작업하고 싶다.
- 프로세스간 데이터 공유가 어렵다
- 컨텍스트 스위칭(Context Switching) 이 무겁다.

이야기를 계속하기 전에 컨텍스트 스위칭(Context Switching)에 대해 이야기 하고 넘어가도록 하죠.

## 2. Context Switching

컨텍스트 스위칭(Context Switching)이란 CPU/코어에서 실행 중이던 프로세스/스레드가 다른 프로세스/스레드로 교체되는 것을 말합니다.

- 컨텍스트(Context) 란?  프로세스/스레드의 상태, CPU, 메모리 등
- 언제? 주어진 시간을 다 사용했거나, IO작업을 해야하거나, 다른 리소스를 기다려야 하거나
- 누구에 의해서 실행(Not Trigger)? OS 커널(Kernel), 커널은 각종 리소스를 관리, 감독 역할을 한다.
- 프로세스들 간의 데이터를 주고 받는 과정: IPC(Inter-process communication)

| -     | Process Context Switching                                                               | Thread Context Switching |
| ---   | ---                                                                                     | ---                      |
| 공통점  | 커널 모드에서 실행, CPU 레지스터 상태를 교체                                                     | 커널 모드에서 실행, CPU 레지스터 상태를 교체 |
| 차이점  | MMU(Memory Manage Unit), TLB(가상 메모리 주소 와 물리 메모리 주소 를 Mapping하는 Cache)를 비워줘야함 | 메모리 관련 처리를 안해서 더 가벼워 |

- 캐시 오염(Cache pollution) → 컨텍스트 스위칭 이후에 성능이 떨어질 때도 있음
- 유저 관점에서 컨텍스트 스위칭 → 순수한(pure) 오버헤드(overhead)

## 3. 스레드(Thread)

<p>
    <img src="/assets/images/post/cs/process-thread/thread.png" width="500" height="400" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

자, 그래서 프로세스의 아쉬운 점을 해결하기 위해 나온 방법이, **스레드(Thread)**, 아래와 같은 특징을 가지죠. 프로세스에서 자원을 공유하여 Contexting Switching의 비용을 줄이는 것 처럼 보이네요!

- 프로세스는 한 개 이상 스레드를 가질 수 있다.
- CPU에서 실행되는 단위(Unit of execution)
- 스레드들은 자신들이 속한 프로세스의 메모리 영역을 공유
- 같은 프로세스 내 스레드끼리 컨텍스트 스위칭이 가볍다.

## 4. 스레드의 종류

가만 보면 스레드는 프로세스를 작은 단위로 더 쪼개놓은 것 같지 않으신가요? 그런 스레드에 종류로는 **하드웨어 스레드, OS 스레드, 네이티브 스레드, 커널 스레드, 유저 스레드, 그린 스레드** 가 있습니다. 

1. 하드웨어 스레드
    - 인텔의 **hyper-threading**: 물리적인 코어마다 하드웨어 스레드가 두 개
    - OS관점에서는 가상의 코어 → 싱글 코어 CPU 하드웨어 스레드 두 개 → OS는 듀얼 코어로 인식

2. OS 스레드(Native Thread, Kernel Thread, Kernel-level Thread, OS-level Thread)
    - OS 커널 레벨에서 생성되고 관리되는 스레드
    - CPU에서 실제로 실행되는 단위
    - CPU 스케줄링의 단위
    - OS 스레드의 컨텍스트 스위칭은 커널 개입(비용 발생)
    - 사용자 코드와 커널 코드 모두 OS 스레드에서 실행

3. Kernel thread
    - OS thread와 동일시 하기도 함
    - OS 커널의 역할을 수행하는 스레드

4. 유저 스레드
    - 프로그래밍 레벨에서 추상화한 것! CPU에서 실행이 되려면 반드시 OS Thread 연결해야한다.
    - OS와는 독립적으로 유저레벨에서 스케줄링 되는 스레드
    - One to One model, race condition 가능성, 멀티코어
    - Many to One model, race condition이 일어날 확률 적음, 멀티코어 X, 한 스레드 블락 → 모든 스레드 블락
    - Many to Many model
    
5. 그린 스레드
    - Java의 초창기 버전에서는 Many to One 스레딩 모델을 **그린 스레드** 라고 함
    - OS와는 독립적으로 유저 레벨에서 스케줄링되는 스레드


여기까지 레퍼런스에 걸린 **[쉬운코드 유튜브 영상](https://www.youtube.com/@ez.)을 보시면 좀 더 이해하기 쉽게 내용을 들으실 수 있습니다.** 그리고 “왜 프로세스에서 스레드까지 필요할까?”에 대해서 흐름이 있지 않나요? 저는 아래와 같은 아이디어일 수 있다고 생각했습니다.

`“Multi-programming → Multi-tasking → Thread → Multi-processing/Multi-threading”`

1. 컴퓨터 작업은 어디서 처리가 돼?(Process)
2. 여러가지 일을 처리하고 싶다! (Multi-programming, Multi-tasking)
3. 프로세스가 하나의 작업을 처리하는 시간이 매우 작다!(Quantom, 여기서 동시성(Concurreny가 생길 수 있겠죠?)
4. 프로세스간에 메모리를 교체하는 Context Switching에 시간을 줄이고 싶어! (여기서 스레드가 나오죠)
5. 좀 더 많을 일을 처리해보자!(Multi-processing, Multi-threading)

사실상 이렇게 이야기하고 보면, Process보다 Thread가 더 작은 단위의 개념이라 하드웨어 입장에서 작업을 처리할 때 Thread를 단위로 처리할 것 같네요. 그리고 재밌는 부분은 그 Thread를 Application, OS(Kernel) 그리고 Hardware로 역할을 구분지어 놓고 하는 것 같죠? Core는 아마도 Hardware 부분일 것 같네요.

## 5. 멀티 프로세스, 멀티 스레드, 멀티 코어
그럼 멀티 프로세스, 멀티 스레드, 그리고 멀티 코어는 이해하기가 좀 더 수월해 집니다.

- **멀티 스레드**: 하나의 프로세스가 동시에 여러 작업을 실행하는데 목적
- **확장된 멀티태스킹**: 여러 프로세스와 여러 스레드가 아주 짧게 쪼개진 cpu time을 나눠 갖는 것
- **멀티프로세싱(multi-processing)**: 두 개 이상의 프로세서나 코어를 활용하는 시스템
- **IO bound, CPU bound**에서 몇 개의 멀티스레드를 하면 좋을까? 이건 [쉬운코드 영상](https://www.youtube.com/watch?v=qnVKEwjG_gM)을 참고해 주시죠!

<p>
    <img src="/assets/images/post/cs/process-thread/burst-time.png" width="500" height="400" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

## 6. Global Interpreter Lock(GIL) in python

<p>
    <img src="/assets/images/post/cs/process-thread/GIL-python.png" width="500" height="300" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   
    
Reference. [https://www.datacamp.com/tutorial/python-global-interpreter-lock](https://www.datacamp.com/tutorial/python-global-interpreter-lock)

저는 Python과 C를 메인으로 쓰고 있어, Python하면 떠오르는 Global Interpreter Lock(GIL)에 대해 조금 더 이야기해보도록 하겠습니다. 먼저 파이썬의 메모리 관리 방식인 Reference Counting이 있기에 GIL이 있을 수 있어, 그 방법에 대해 다뤄보도록 하죠.

- **Reference Counting**: 파이썬이 메모리를 관리하는 방식으로 모든 객체를 카운트하며 그 객체가 참조될 떄 증가/참조가 삭제되면 감소하는 방식으로 동작합니다. count가 0이 되면 삭제 대상이 되며 삭제 cycle이 되면 메모리 할당이 해제됩니다.

- 그렇기 때문에 Thread에서 다른 Thread로 넘어 갈시 GIL을 release/acquire하는 비용이 커서, 저는 Python하면 Multi-processing을 주로 사용하죠. [PyCon US 2021에서 발표한 GIL 메인 컨트리뷰터 나동희님](https://www.youtube.com/watch?v=V18ceQO_JkM)은 “Python이나 Java는 자동으로 메모리 관리를 하는데, Python의 경우 **Reference Counting 방법이 GIL**을 필요로하기 때문에 GIL자체가 사라지는 것은 아직은 아니다.”고 언급하셨습니다.

- 그럼 언제 Thread를 써야돼? I/O bound 작업이 많아 Thread가 대기해야하는 경우에 사용하면 될 것 같은데, 다른 경우도 있을까요?

여기까지 프로세스, 스레드 그리고 코어에 대해서 알아봤습니다. 다음번에는 멀티 프로세스, 스레드시에 생기는 Race Condition과 그에 대한 종류, 해결방안, 그리고 CPU 스케줄링에 대해 다뤄보도록 하겠습니다. (그리고 커널쪽에 가깝게 다가갈 수 있는 임베디드 특징상 프로세스와 스레드를 다루는 예제 코드도 추가해보록 할게요!)

## 6. Reference
- 배민 테코톡(스레드와 프로세스) [1](https://www.youtube.com/watch?v=1grtWKqTn50), [2](https://www.youtube.com/watch?v=LLiV5Yz1AWg), [3](https://www.youtube.com/watch?v=DmZnOg5Ced8)

- 쉬운코드
    - [프로세스, 쓰레드, 멀티태스킹, 멀티쓰레드, 멀티 프로세스](https://www.youtube.com/watch?v=QmtYKZC0lMU)
    - [컨텍스트 스위칭](https://www.youtube.com/watch?v=Xh9Nt7y07FE&t=459s)
    - [쓰레드의 종류](https://www.youtube.com/watch?v=vorIqiLM7jc)
    - [쓰레드 풀](https://www.youtube.com/watch?v=B4Of4UgLfWc)
    - [레이스 컨디션과 동기화](https://www.youtube.com/watch?v=gTkvX2Awj6g)
    - [IO/CPU bound](https://www.youtube.com/watch?v=qnVKEwjG_gM)

- [IO/CPU burst time](https://www.cs.uic.edu/~jbell/CourseNotes/OperatingSystems/6_CPU_Scheduling.html)

- [Python Global Interpreter Lock](https://www.datacamp.com/tutorial/python-global-interpreter-lock)