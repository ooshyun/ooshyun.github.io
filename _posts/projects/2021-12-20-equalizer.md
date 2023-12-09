---
title: Equalizer Design using Digital Filter
key: 20211220
tags: Projects
comment: false
---

<p>
    <img src="/assets/images/project/cascade-parellel-iir.png"> 
    <p align="center">
    <em> Structure of cascaded IIR filters </em>
    </p>
</p>

To implement digital filter in embedded device, this project is designing digital filter in Python and C for simmulation. Since the limitation of the edge device has a small computation capacity, it needs the filter application for adjusting the low-frequency band significantly below 200Hz. As a personal research, the inspiration is from a simple parallel or cascaded Infinite Impulse Response(IIR) filter <a href="https://www.aes.org/e-lib/browse.cfm?elib=19355">[1]</a>, <a href="https://ieeexplore.ieee.org/abstract/document/6891289/">[2]</a>. The process using equalizer simulates in filtfilt and filt function in scipy.signal library also for implementing those functions.
<br>

<!-- {% include image.html 
url="/assets/images/project/cascade-parellel-iir.png" 
custom__conf="projects__img__center"
%} -->

<p>
    <p align="center">
        <img src="/assets/images/project/transposed-direct-form-II.png" width="45%" height="45%"> 
    </p>
    <p align="center">
    <em> Transposed direct form II </em>
    </p>
</p>

Specifically, I designed a biquid filter for high-pass, low-pass, band-pass, shelf, peak, and notch forms. And also it tested parallel IIR filter application with transposed direct form II. The grahpical equalizer included wrapped fixed pole design and minimum phase through Hilbert Transform. The code for those designs and test is on the <a href="https://github.com/ooshyun/FilterDesign">github</a>. The code includes not only the implementation of the filter but also filt function and several digital signal processing examples such as <a href="https://github.com/ooshyun/FilterDesign/tree/master/study/fft_scratch">vanilia fft</a>. The part of this research implements into Olive Max and Olive Pro project by converting C and Embedded C for Hifi2-mini.

**Measurement**
<!-- {% include image.html 
url="/assets/images/project/graphical-eq.png" 
custom__conf="projects__img__center"
%} -->
<p>
    <img src="/assets/images/project/graphical-eq.png"> 
    <p align="center">
    <em> Measurement of Graphical Equalizer </em>
    </p>
</p>

The below list of details contains how to design and simulate the digitial filter and the implmentation of graphical equalizer from a paper "Efficient design of a parallel graphic equalizer, 2017".

**Applications**
- Biquid filter
- Cascaded IIR filter
- Parallel IIR filter
    - Wrapped fixed poled frequency weighting
    - Cubic Hermite and spline interpolation
    - Minimum phase system using Hilbert Transform
    - Least square solution

**Code**
- <a href="https://github.com/ooshyun/FilterDesign">Digital Filter design</a>


**Reference** <br>
[1] Bank, Balázs, Jose A. Belloch, and Vesa Välimäki. "Efficient design of a parallel graphic equalizer." Journal of the Audio Engineering Society 65.10 (2017): 817-825.<br>
[2] Rämö, Jussi, Vesa Välimäki, and Balázs Bank. "High-precision parallel graphic equalizer." IEEE/ACM Transactions on Audio, Speech, and Language Processing 22.12 (2014): 1894-1904.<br>