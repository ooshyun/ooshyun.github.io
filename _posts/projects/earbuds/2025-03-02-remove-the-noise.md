---
title: Remove the noise in a hearing aid, part 1
key: 20250302
tags: device TinyML SpeechEnhancement SourceSeparation
---
시작은 2022년 3월 14일이었습니다. "승현님, 저희 소음제거 알고리즘을 개선시켜주세요!" 

저는 보청기 스타트업에서 일하는 임베디드 DSP 엔지니어 였죠. 이 업무는 보청기로 들어오는 소리를 마이크로부터 데이터로 받아, 이용자에게 맞는 소리로 이 신호를 적절하게 처리해서 스피커로 내보내는 알고리즘을 설계하는 일입니다. 에어팟 프로나 갤럭시 버즈에서 보이는 Equalizer, 음량조절, 외부소리 듣기, 노이즈캔슬링과 같은 알고리즘이 제 파트에서 핸들링돼서 나가는 일이죠. 기존에 다른 시스템보다 훨씬 하드웨어 스펙이 열등하면서 (e.g. arm cortex-m3) 배터리 용량도 중요하기 때문에 연산하나하나가 세심하게 다뤄져야 했습니다. 그러던 중에 업무가 하나 떨어진거죠.

<!--more-->

## Prologue

소음제거(Noise Reduction)이라는 연구는 Speech Enhancement(음질 개선)이나 Source Separation(음원 분리) 기술을 많이 이용하곤 합니다. 이 연구는 Hearing aid에서는 마이크로 들어오는 데이터에서 잡음을 없애면서 사람의 음성을 선명하게 만들면서 증폭시키기 위해서 사용했습니다. 연구 방향성은 크게 두 갈래로 나눌 수 있었는데, 기존에 Classical Statistical method을 이용할 것인가, 아니면 Machine Learning 을 이용할 것인가로 나뉩니다. 당시 조사했던 대로라면 아래와 같이 나눌 수 있었습니다.

```
ML Method
    Magnitude spectrum
        spectral masking
        spectral mapping
    Complex domain
    Time domian
    Multi-stage
    Feature augmentation
    Audio-Visual SE
    Netword design
        Filter design
        Fusion technique
        Attention
        U-Net
        GAN
        Auto-Encoder
        Hybrid SE
        SepFormer(Transformer)
    Phase reconstruction
    Learning strategy
        Loss function
        Multi-task learning
        Curriculum learning
        Transfer learning
    Model Compression
    
Classical
    Wiener filter
    Kalman filter
    minimum variance distortionless response (MVDR) beamformer
```

방향성을 선택하기에 앞서 제가 가진 점과 회사 인프라, 그리고 현재 연구 진행정도와 미래에 대한 전망을 가늠했습니다. 우선, 저는 ML, AI, DSP에 기본적인 knowledge을 가지고 있었고, DSP 어플리케이션(e.g. all types of digital filters, equalizer) 을 설계할 수 있고, 회로를 읽을 수 있었습니다. AI로 방향을 하자고 생각하니, 회사내에 인프라나 관련 연구에 관심이 있는 동료가 없거니와, 결국 최종적인 형태로 AI를 TinyML로써 Edge device에 올려야 하는데, 이를 하는 연구는 큰 커뮤니티가 보이지 않았습니다. 커뮤니티 중 제일 컸던 [tflite-micro](https://github.com/tensorflow/tflite-micro) 에서도 아직 under-develop된 경우가 많아서 '과연 현실 가능성이 있는 연구인가?'에 대한 불확실성이 있었습니다(현재는 Torch 2.0 이후로 Torch에서도 해당 연구가 활발하게 이뤄지고 있습니다). 

1. 파트너로 협업하는 회사에서 Neural Network에 대한 알고리즘을 옵션으로 넣는데, 기초적인 내용으로 가늠된다.

2. [Bose에서 보청기를 위해 연구된 Speech Enhancement 연구](https://arxiv.org/abs/2005.11138) 가 2020.05에 발표됐고 이 연구에 사용된 모델(LSTM)은 가장 기본이 되는 연구다.

그럼에도 앞선 두 가지 단서를 보고 TinyML로 Speech Enhancement 분야가 가능성은 존재하고, 현재는 기초 단계에 있지만 그러기에 연구에 시간을 쏟는게 그만큼에 가치를 가질 수 있다 보였습니다. 또한 회로를 읽을 줄 아는 부분에 있어서 하드웨어와 밀접한 연관이 있는 연구가 더 메리트가 있다고 생각했습니다. 그래서 TinyML과 Speech Enhancment 연구를 이 때부터 시작합니다. 먼저 Task인 Speech Enhancement 부터 말이죠.

Speech Enhancement 연구는 데이터, 모델, 그리고 평가방법으로 구성됩니다. 우선 데이터를 크게 두 가지로 나눠어 있습니다. 타겟이 되는 소리와 소음이 되는소리 입니다. 이를 위한 데이터 셋을 만든 연구는 "어떤 소리를, 그리고 어떻게 소리를 합칠 것인가?" 에 대해서 고민합니다. 그리고 모델을 디자인하는 사람들은 이 데이터셋을 이용해서 합쳐진 소리를 타겟 소리로 만드는 모델을 만들죠. 마지막으로 이를 평가하는 방밥을 고민하는 연구가 있습니다. 남은 내용은 Speech Enhancement를 풀기 위해 고민한 세 가지에 대해서 이야기해보겠습니다.

- Which dataset we can use?
- Which model backbone we can use?
- How can we evaluate the model(+Loss function)?

**** 나머지 부분은 신호처리에 대한 지식이 없으면 이해하기 어려울 수 있으니 참고 바랍니다**

## Which model backbone we can use?
모델은 입력 데이터 형태에 따라 크게 두 가지 방향으로 나뉩니다.

1. Time domain
2. Frequency domain

제가 연구를 했던 당시(2022.03)는 Frequency domain(Frequency)에서 연구가 압도적으로 많았었는데, Time domain에서 유명한 모델인 WavUnet, ConvTasNet이 있기 때문에 양쪽 모두 고려를 했어야 했습니다. 또한 제가 타겟하는 하드웨어 플랫폼이 Hearing aid인점을 고려해 Bose의 연구를 레퍼런스 삼아 모델 크기를 1MB를 기준으로 가능성을 가늠했습니다. 

후보는 기본적인 Neural Network에서 평가, Bose 연구를 벤치마킹할 수 있는 RNN계열, Frequency domain에서 대표적으로 알려진 CRN, DCCRN, Time domain에서 대표적인 WavUnet, ConvTasNet, 그리고 아직 실험중인 Transformer형태의 Sepformer 까지 총 10가지 모델을 평가했습니다. 언급되지 않은 모델은 앞선 모델들에서 좀 더 디벨롭된 모델에서, 오픈소스로 구하기 쉬운 쪽으로 선택했습니다.

1. DNN(Deep Neural Network)
2. RNN(Recurrent Neural Network)
3. CRN(Convolutional Recurrent Network)
4. DCCRN(Deep Complex Convolutional Recurrent Network)
5. Unet
6. WavUnet
7. DCUnet
8. Demucs-v2
9. ConvTasNet
10. Sepformer

해당 모델들에 대한 자세한 코드는 [해당 링크](https://github.com/ooshyun/Speech-Enhancement-Pytorch)를 확인해 보시면 좋을 듯 싶네요. 참고로 테스트를 실시간으로 확인해보고 싶으시면 Framework로는 [Facebook에서 연구하는 플랫폼](https://github.com/facebookresearch/denoiser)을 이용하시면 실시간으로 알고리즘이 적용된 소리를 들어 볼 수 있었습니다. 저 같은 경우는 노트북에 마이크로 들어오는 데이터를 실시간으로 이어폰으로 들어 볼 수 있었습니다.

## Which dataset we can use?
다음은 데이터셋을 어떤 것을 쓸 것인가? 입니다. 음원에 관련된 데이터 셋은 참 많습니다. [이 링크](https://github.com/ooshyun/Docs-for-SpeechEnhancement-and-TinyML/blob/master/docs/dataset_sound.md)에 정리를 해놨으니 참고하시면 좋을 듯 싶네요.

사실, 여기선 제게 선택지의 폭은 넓지 않았습니다. 사내에서 관련한 연구 리소스가 없었거니와, 기업에게 라이센스가 오픈인 경우는 VoiceBank-DEMAND 라는 데이터 셋밖에 없었거든요. 이 데이터셋은 VoiceBank라는 사람 목소리 데이터셋에 DEMAND라는 소음 데이터셋을 합친 데이터입니다. 그래서 한 가지 더 생각해봅니다. Speech Enhancement Challenge는 없을까?

1. Microsoft DNS Challenge
2. Clarity Challenge

챌린지는 위에 두 가지가 있습니다. Microsoft의 경우는 뒤늦게 발견해서 이용하지는 못했으나, 데이터 셋이 큰 편에 속해서 관심있으신 분은 해당 챌린지를 살펴보시면 좋을 듯 싶네요. Clarity Challenge의 경우는 Hearing Test 데이터, 소음과 목소리가 합쳐진 데이터, 목소리 데이터, 그리고 데이터를 수집한 시나리오를 제공해줍니다. 결론적으로 저는 Clarity Challenge + VoiceBankDEMAND, 두 가지 데이터 셋 조합으로 모델 훈련을 진행합니다. 

## How can we evalueate the model?
모델의 성능을 가장 좋게 평가할 수 있는 방법은 바로 사람이 듣는 거겠죠. 하지만 음원마다 길이가 꽤 길기 때문에 이를 일일이 저희가 들어보기는 현실적으로 어려울 겁니다. 그래서 연구자들은 몇 가지 Metric을 내놓습니다. 그 중 대표적인 게 STOI(Short-Time Objective Intelligibility), PESQ(Perceptual Evaluation of Speech Quality), SDR(Signal Distortion Ratio), SI-SDR(Scale-Invariant Signal Distortion Ratio) 입니다. 관련해서 설명하기에는 내용이 길어, 가장 직관적인 벡터의 유사성으로 두 개의 신호를 평가한 SI-SDR을 가장 먼저 평가하고 나머지 Metric들은 참고용으로 살펴봤습니다. 자세한 내용과 테스트는 [이 링크](https://github.com/ooshyun/Speech-evaluation-methods)를 참고하셔도 좋을 듯 싶어요.

앞서 설명한 내용은 저희가 결과로 나온 음원 중 어떤 음원이 원하는 방향에 맞는 음원인가에 대한 내용이었습니다. 그럼 마지막으로 남은 loss function은 어떻게 고를 수 있을까요? 이건 time, frequence domain에 대한 내용을 아시는 분이면 쉽게 이해하실 수 있을 겁니다. 바로 모델의 출력이 저희가 원하는 타겟의 소리 time domain이나 frequency domain에서 값들괴 같으면 됩니다. 다만 frequency domain일 경우는 amplitude 만을 고려하는 방법도 있고, phase를 함께 고려하는 방법도 있습니다. 그리고 한 가지 더, 저는 SI-SDR이 이와 유사하다고 판단해 이를 loss function으로 하는 실험도 함께 진행했습니다.

*다만 Source separation의 경우에는 여러명의 화자를 타겟으로 하는 경우도 있어, 이 때는 나오는 모델 결과도 화자 수 만큼 나와서 이를 타겟 음원과 Permutation을 돌려 loss가 최대인 경우를 loss로 선택합니다.

## Epilogue
위 실험은 [여기서](https://github.com/ooshyun/Speech-Enhancement-Pytorch/tree/master) 보실 수 있다시피 특정 음원에 대해서는 소음을 잘 제거했지만 모든 음원에 대해서는 그렇지는 못했습니다. 원하는 모델의 성능까지 달성하지는 않았지만, 해당 실험을 진행하면서 운좋게 Clarity Challenge에서 Top5도 들 수 있었습니다. 덕분에 나름에 원동력도 가지고, 모델의 성능에 초점을 맞춰 더 디벨롭 시키는 방향도 있지만 우선은 "모델을 Edge device에 돌리자!" 라는 파이프 라인이 우선적이라고 판단해 해당 연구는 여기까지 진행했습니다.

<p>
    <img src="/assets/images/post/device/earbud/clarity_challenge.png" width="500" height="300" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption"> Top five in Clarity ICASSP Grand Challenge  </em>
    </p>
</p>

이 자료가 관련 연구를 하시는 분들께 도움이 되길 바라며, 저는 Part 2. Platform for microphone streaming 으로 다시 돌아오겠습니다.

## Reference

### 1. ML models Platform 
- [https://github.com/asteroid-team/asteroid](https://github.com/asteroid-team/asteroid)
- [https://github.com/speechbrain/speechbrain](https://github.com/speechbrain/speechbrain)
- [https://github.com/facebookresearch/demucs](https://github.com/facebookresearch/demucs)

### 2. Other Research for speech enhancment methods 
- [https://github.com/Wenzhe-Liu/awesome-speech-enhancement](https://github.com/Wenzhe-Liu/awesome-speech-enhancement/blob/master/README.md)
- [https://github.com/nanahou/Awesome-Speech-Enhancement](https://github.com/nanahou/Awesome-Speech-Enhancement/blob/master/README.md)
- [https://ccrma.stanford.edu/~njb/teaching/sstutorial/part1.pdf](https://ccrma.stanford.edu/~njb/teaching/sstutorial/part1.pdf)
- [https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/interspeech-tutorial-2015-lideng-sept6a.pdf](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/interspeech-tutorial-2015-lideng-sept6a.pdf)
- [https://www.citi.sinica.edu.tw/papers/yu.tsao/7463-F.pdf](https://www.citi.sinica.edu.tw/papers/yu.tsao/7463-F.pdf)