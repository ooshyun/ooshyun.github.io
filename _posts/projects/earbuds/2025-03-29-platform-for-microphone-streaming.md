---
title: Platform for microphone streaming, part 2
key: 20250329
tags: TinyML SpeechEnhancement SourceSeparation STM32
---
다음 단계로 Speech enhancment 모델을 테스트하기 위해서 임베디드 플랫폼을 선정해야 했습니다. Web, App, Desktop based 어플리케이션은 [Denoiser](https://github.com/facebookresearch/denoiser)가 있어 쉽게 모델을 실험해 볼 수 있었지만, 임베디드 시스템에서는 DSP 보드에서만 알고리즘을 설계해봤기 떄문에, 이 플렛폼에 모델을 테스트하기 위해 달성할 수 있는 목표로 두 가지 정했습니다.

<p>
    <img src="/assets/images/post/device/earbud/platform-for-microphone-streaming/streaming_platform.png" width="400" height="200" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"> TODO </em>
    </p>
</p>

첫 번째는 마이크 스트리밍에 필요한 마이크에 해당하는 커널을 조작할 수 있는 것과 이를 스트리밍으로 불러와서 스피커나 파일형태로 내보낼 수 있는 플랫폼입니다. 그리고 다른 하나는 모델을 변환하고 이를 low-level programming language로 포팅하는 방법울 익히는 것 입니다. 간단하게 말하면 Tensorflow Lite C API를 사용하는 것인데, 조금 더 최적화에 초점을 맞춰 이를 위해 [TinyML](https://hanlab.mit.edu/courses/2024-fall-65940)에서 Object detection이라는 더 잘 알려진 테스트의 모델을 이용해서 포팅하는 과정까지 진행하기로 했습니다. 후자의 경우는 [이전에 포스팅한 글](https://ooshyun.github.io/2023/12/16/Optimization-for-tiny-engine-2.html)이 있어 참고하시면 되겠습니다.

## Platform
플랫폼을 제공할 수 있는 회사들은 SoC를 제조해서 이를 PCB형태로 테스트보드 형태의 제품이 있고, 이를 제품의 형태로 만들 수 있었습니다. 
그래서 MCU 단계에서 시용할 수 있는 ARMv7-M CPU 라인업에서 Bose 선행연구에 선행연구나 TinyML 강의에서 사용하는 플랫폼이 따라가기 수월해 STM32사에 STM32F746G-DISCOVERY 보드를 선택해서 진행했습니다.

- ST Microelectronics
- TI(Texas Instruments)
- Nordic Semi
- Silocon Labs
- Sytiant
- Green Wave Technologies
- Raspberry Pi
- Arduino
- RISC-V
- Espressif
- SparkFun
- Eta Compute
- OpenMV
- Sony
- Bitcrase AdduoCam
- XMOS
- Mbed
- Renesasa Electronics
- Microchp Technology
- NXP Semiconductors

## Designs streaming application in STM32F746G-DISCOVERY
STM32를 처음 사용하면서 아래에 과정대로 프로팅타이핑까지 진행했습니다. 다뤄야 할 기능(USB, FIR, FFT, Microphone, Streaming)에 대한 예제를 찾아 분석하고 해당 예제에 맞게 전체 파이프라인을 개발했습니다. 참고로 예제는 STM32CubeIDE를 설치하면 해당 보드에 맞는 예제들을 함께 제공해줍니다. 참고로 IDE는 보드에 burn하는 과정인 compile and write the flash in device(.elf)를 편하게 할 수 있도록 도와주기 때문에 회사에서 지원이 있다면 Keil이라는 IDE가 편리하고, 외에는 개발하는 플랫폼의 IDE를 사용하는 것이 편리함에서나 추후에 디버깅, 프로파일링에서 좋습니다.

1. 예제를 구한다.
    - [USB Full speed to speaker](https://github.com/STMicroelectronics/STM32CubeH7/blob/master/Projects/STM32H743I-EVAL/Applications/USB_Host/AUDIO_Standalone/readme.txt)
    - [FFT, FIR Demo](https://www.st.com/en/embedded-software/x-cube-dspdemo.html)
    - [Mic streaming and FFT]( https://github.com/jhang-jhe-wei/Fast-Fourier-transform-using-microphone-in-STM32f746g-DISCOVERY)
    - [Mic Streaming and Recording to USB](https://github.com/ada-sound/X-CUBE-USB-AUDIO)
    - [Lecture for Audio Loop in STM32F769I-Discovery](https://github.com/ProjectsByJRP/audio_pass-through)
        - [설명 영상](https://www.youtube.com/watch?v=O2XaCFsWxSw&t=1401s)

2. 전체 파이프라인
    - Streaming Pipe line(Encoder, Decoder)				
        - OS for Embedded MCU (RTOS/FreeRTOS or single task/infinite loop)		
        - Mems microphone
        - Wav file load					
        - Speaker					
        - Wav file save
    - DSP Pipeline: FFT
    - Model Pipeline	
        - Port Tensorflow Lite 
        - Compression Technique: tflite conversion

## Conclusion
이번 과정에서는 자세한 내용을 다루기보다는 모델을 테스트하기 위한 임베디드 플랫폼을 선정하고, 이를 위한 예제로 필요한 기능을 분석하고, 전체 파이프라인을 설계하는 과정을 다뤘습니다. 하면서 보드를 사용하는 것보다 이어폰과 같이 제품의 형태로 만들어보면서 모델 성능을 실험해보면 좋을 것이라고 생각해서, 다음 단계 한, 이를 위한 PCB를 디자인하고, 이를 제품의 형태로 만들어보는 과정에 대해서 다뤄보겠습니다.