---
title: Preproces, Compile and Linker
aside:
    toc: true
key: 20230602
tags: CS
---

오늘 시간에는 프로그램이 컴파일 돼 실행되는 과정을 이야기 해보자 합니다. 그리고 언어 중에서는 제가 사용하는 C와 스터디에서 다른 분들이 많이 사용하실 것같아 Java도 함께 곁들여 진행해보겠습니다. 혹시나 틀린 부분이 있다면 콕콕 집어주시기 바랍니다! 진행은 요 [링크](https://github.com/VSFe/Tech-Interview/blob/main/02-OPERATING_SYSTEM.md)에 질문들을 따라가보면서 살을 덧붙여볼게요.

- 링커와, 로더의 차이에 대해 설명해 주세요.
- 컴파일 언어와 인터프리터 언어의 차이에 대해 설명해 주세요.
- JIT(Just-in-time) 에 대해 설명해 주세요.
- 본인이 사용하는 언어는, 어떤식으로 컴파일 및 실행되는지 설명해 주세요.

<!--more-->

**컴파일 과정하면 어떤 생각이 드시나요?** 저는 가장 간단하게 **“번역”** 이라고 생각이 드는 데요, 0과 1로만 만들어진 기계언어를 사람이 코드를 통해 조작하기 위해 코드를 기계어 번역해서 실행파일과 연결하는 과정이라고 봅니다. 그래서 그 대표적인 Compile 언어로는 C와 Java를 가져와봤습니다(혹시 Java에 대해서 틀린 부분이 있으면 바로바로 집어주시기 바랍니다!).

## 1. C vs Java

컴파일 과정에 앞서 **C와 Java의 차이점**에 대해서 조금 언급하고 넘어 가야겠네요. 기본적으로 C가 Java보다 빠르다고 하는데요, 이야기를 들어보면 이해가 가실거에요. 

> **둘의 차이점중에 가장 먼저 보이는 것은 “절차 지향적”이냐, “객체 지향적”이냐의 차이점입니다.**
> 

Java의 경우 Java Virtual Machine(JVM)가 코드를 기계어(bytecode를 의미하겠죠?)로 변환시키고 실행시킨다고 합니다. 그래서 Just in time(JIT) Compiliation이라고 많이 들어보셨을 텐데, 변환과 실행이 그 때 그때 된다는 의미겠죠? 그럼, 왜 Java가 객체지향적이라고 말하는 지 이해가 가시나요?

반면 C의 경우 실행파일로 들어간 코드 중 main 이라고 선언된 함수내에서 돌아가는 하나의 루프가 실행의 전부입니다(main말고 이름을 바꿀 수도 있어요!). 간단하죠? 대신 Java 보다 코드 자체를 직관적으로 이해하는 데 힘들다고 합니다. 그리고 “Preprocess(전처리)”라는 단계를 거쳐서 Compilation을 합니다. 이 전처리과정은 이후에 설명드릴 텐데, 코드에서 필요없는 부분을 싹- 빼고 기계어로 번역하신다고 생각하면 됩니다.

이외에도 많은 차이점이 있는데, 오늘은 컴파일 과정이 주제라서 여기까지만 이야기해 볼게요. 혹시 다른 차이점 아시는 게 있나요? 말해주세요!

들어가기에 앞서 본 내용은 [이 링크](https://www.geeksforgeeks.org/cc-preprocessors/)를 참조한 것이므로, 본문을 원하신다면 가보시길 추천드립니다. 그럼 본격적으로 컴파일 과정을 들어가볼게요. Java는 Preprocess(전처리) 과정을 빼면되니 참고하시길 바랍니다. 아래의 그림은 C언어가 기계어로 번역해, 실행 파일(어플리케이션)과 연결하는 과정을 보여 줍니다. 

<p>
    <img src="/assets/images/post/cs/compile/procedure.png" width="200" height="500" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"> Reference. https://www.geeksforgeeks.org/cc-preprocessors/</em>
    </p>
</p>   

한 번 예제로 따라가 볼까요?

```c
#include<stdio.h>
#define add(a, b) (a+b) // using macros
int main()
{
	int a=5, b=4;
	printf("Addition is: %d\n", add(a, b));
	return 0;
}
```

이런 C 코드가 `gcc -Wall example.c –o example` 요 명령어를 통해 `example.exe` 로 변하는 과정을  볼겁니다. 

- 참고  `gcc` 의 경우 c 언어용 컴파일러, `-Wall`은 그 컴파일러에서 옵션입니다. 그리고 실습을 해보고 싶으신 분들을 위해, 중간 과정(Compilation, Assembly)에서 나오는 파일들을 보고 싶으시다면 `gcc -Wall -save-temps example.c –o example` 이렇게 쓰시면 됩니다

과정은 `Pre-processing → Compilation → Assembly → Linking` 요롷게 진행될 예정입니다. 

## 2. Preprocess in C

### 2.1 Preprocess

Preprocess(전처리) 과정은 Java에는 없고 C만 가지는 부분입니다. 이 과정은 크게 네 가지 역할을 하고나서 처리한 코드는 `example.i` 라는 파일로 나옵니다.

- Removal of Comments
- Expansion of Macros: 매크로가 뭐냐? 위에 예시를 참고해주세요!
- Expansion of the included files.
- Conditional compilation

### 2.2 Compile

다음 단계는 컴파일단계 입니다. 이 과정을 거친 코드는 `example.s` 로 나오고, 어셈블리 레벨의 명령어들로 아래처럼 번역이 됩니다.

```nasm
.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0	sdk_version 12, 3
	.globl	_main                           ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	subq	$16, %rsp
	movl	$0, -4(%rbp)
	movl	$5, -8(%rbp)
	movl	$4, -12(%rbp)
	movl	-8(%rbp), %esi
	addl	-12(%rbp), %esi
	leaq	L_.str(%rip), %rdi
	movb	$0, %al
	callq	_printf
	xorl	%eax, %eax
	addq	$16, %rsp
	popq	%rbp
	retq
	.cfi_endproc
                                        ## -- End function
	.section	__TEXT,__cstring,cstring_literals
L_.str:                                 ## @.str
	.asciz	"Addition is: %d\n"

.subsections_via_symbols
```

### 2.3 Assembly

그럼 위에 저 어셈블리 언어를 위한 명령어들을 “어셈블러”가 또 바꿔줘야 겠죠? 그 과정을 거친 파일은 `example.o` 형태를 가집니다. 이 과정에서 코드는 이제 보고 알아볼 수는 없습니다…

아 이렇게는 잘 안보구요.

```bash
Ïúíþ^G^@^@^A^C^@^@^@^A^@^@^@^D^@^@^@^H^B^@^@^@ ^@^@^@^@^@^@^Y^@^@^@<88>^A^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@°^@^@^@^@^@^@^@(^B^@^@^@^@^@^@°^@^@^@^@^@^@^@^G^@^@^@^G^@^@^@^D^@^@^@^@^@^@^@__text^@^@^@^@^@^@^@^@^@^@__TEXT^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@9^@^@^@^@^@^@^@(^B^@^@^D^@^@^@Ø^B^@^@^B^@^@^@^@^D^@<80>^@^@^@^@^@^@^@^@^@^@^@^@__cstring^@^@^@^@^@^@^@__TEXT^@^@^@^@^@^@^@^@^@^@9^@^@^@^@^@^@^@^Q^@^@^@^@^@^@^@a^B^@^@^@^@^@^@^@^@^@^@^@^@^@^@^B^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@__compact_unwind__LD^@^@^@^@^@^@^@^@^@^@^@^@P^@^@^@^@^@^@^@ ^@^@^@^@^@^@^@x^B^@^@^C^@^@^@è^B^@^@^A^@^@^@^@^@^@^B^@^@^@^@^@^@^@^@^@^@^@^@__eh_frame^@^@^@^@^@^@__TEXT^@^@^@^@^@^@^@^@^@^@p^@^@^@^@^@^@^@@^@^@^@^@^@^@^@<98>^B^@^@^C^@^@^@^@^@^@^@^@^@^@^@^K^@^@h^@^@^@^@^@^@^@^@^@^@^@^@2^@^@^@^X^@^@^@^A^@^@^@^@^@^L^@^@^C^L^@^@^@^@^@^B^@^@^@^X^@^@^@ð^B^@^@^B^@^@^@^P^C^@^@^P^@^@^@^K^@^@^@P^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^A^@^@^@^A^@^@^@^A^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@UH<89>åH<83>ì^PÇEü^@^@^@^@ÇEø^E^@^@^@ÇEô^D^@^@^@<8b>uø^CuôH<8d>=^O^@^@^@°^@è^@^@^@^@1ÀH<83>Ä^P]ÃAddition is: %d
^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@9^@^@^@^@^@^@^A^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^@^T^@^@^@^@^@^@^@^AzR^@^Ax^P^A^P^L^G^H<90>^A^@^@$^@^@^@^\^@^@^@pÿÿÿÿÿÿÿ9^@^@^@^@^@^@^@^@A^N^P<86>^BC^M^F^@^@^@^@^@^@^@-^@^@^@^A^@^@-&^@^@^@^B^@^@^U^@^@^@^@^A^@^@^F^A^@^@^@^O^A^@^@^@^@^@^@^@^@^@^@^G^@^@^@^A^@^@^@^@^@^@^@^@^@^@^@^@_main^@_printf^@^@
```

Hex형태로 보면 요렇게는 봅니다. 분석을 한다면 어디에 사용할 수 있을까요? (근데 중간에 printf 와 같은 함수는 남아있는게 보이시나요?)

<p>
    <img src="/assets/images/post/cs/compile/assembly.png" width="200" height="500" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

### 2.4 Link


<p>
    <img src="/assets/images/post/cs/compile/link.png" width="200" height="300" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"></em>
    </p>
</p>   

짠! 이제 실행파일(어플리케이션)을 만들어야죠? 마지막 단계는 위에 번역된 모—든 기계어를 실행파일과 연결시켜야겠죠? Link과정에서 추가적으로 프로그램을 OS에서 실행할 수 있게 프로그램의 시작과 끝을 조금 추가해주고 나머지 순서는 main 함수(개발자가 지정한 함수, default는 main)을 따라갑니다.

이 과정을 끝내기 전 example.o 와 example.exe를 **`$size example.o`** and **`$size example`** 를 쓰시면 아래 처럼 실행파일 안에 코드 영역별로 차지하고 있는 부분을 볼 수 있습니다. 

```bash
(mlp) seunghyunoh@Seunghyuns-MacBook-Air ctest % size test_compile.o
__TEXT  __DATA  __OBJC  others  dec     hex
138     0       0       32      170     aa
(mlp) seunghyunoh@Seunghyuns-MacBook-Air ctest % size test_compile  
__TEXT  __DATA  __OBJC  others  dec     hex
16384   16384   0       4295000064      4295032832      100010000
```

실행파일은 부수적으로 더 많이 들어가 있죠? __TEXT, __DATA,  __OBJC,  others,  dec,  hex 는 그럼 각기 뭘까요? 메모리 영역을 아신다면 유추가 가능할 겁니다. 궁금하면, [여기](https://www.notion.so/C-C-Memory-map-cbed13ab87514deca16ece6a0417e0af?pvs=21)에 사진을 참고해보세요!

### 2.5 Load

흠흠, 근데 “로더”라는 친구가 아직 설명이 되지 않았네요. 프로그램도 만들어졌다, 그리고 각 메모리 영역에 들어가야 할 내용과 어플리케이션의 순서도 정해졌다, 그럼 남은 건 OS에서 실행하는 일만 남았겠죠? 이건 유추해보시길 바랍니다.

여기까지 프로그래밍 언어가 컴퓨터에서 실행되기 위한 과정을 살펴봤습니다. 조금 과정이 머릿속에 그려지시나요? 설명드린 내용은 정말 간략한 내용이구요, 번역과정은 빠르면 빠를 수록 좋잖아요? 이 과정 중에 어셈블리어 번역을 좀 더 빠르게 하기 위해 `.i` 대신 **LLVM 컴파일러**를 사용하기 위해 `.ll` 로 바꾸는 과정등 제사하게 들어가면 더 이야기가 많은데, 그건 다음번에 제가 사용하게 된다면 더 풀어보도록 하겠습니다.