---
title: Olive Pro and Olive Max, Earbuds for hearing aid
key: 20210413
tags: Projects
---
<!-- <div class="projects__article__right">
{% include image.html url="/assets/images/project/img_olivepro.jpg"  
%}
</div> -->
In Olive Union, who designed, manufactured, and serviced the application for a hearing aid, I contributed to <a href="https://www.indiegogo.com/projects/olive-pro-2-in-1-hearing-aids-bluetooth-earbuds#/">Olive Pro</a> and <a href="https://www.indiegogo.com/projects/olivemax-3-in-1-hearing-aid-earbud-tinnitus-app#/">Olive Max</a> as a digital signal processing engineer in embedded systems. 

<p>
    <img src="/assets/images/project/img_olivepro.jpg"> 
    <p align="center">
    <em> The product, Olive Pro </em>
    </p>
</p>

In the hearing aid open-source platform, <a href="https://github.com/claritychallenge/clarity">the clarity challenge</a> is continuing to research for enhancing speech clarity even if it amplifies most of the sounds. As a reference of this platform, I was trying to contribute to exploring the products and algorithms of hearing aid.
<br>

<p>
    <p align="center">
        <img src="/assets/images/project/img_olivemax_small.png" height="50%" width="50%"> 
    </p>
    <p align="center">
        <em> The product, Olive Max </em>
    </p>
</p>

We release two products, Olive Pro and Olive Max, whose final can cover the severe hearing loss. As a detail, we designed the real-time processing pipeline, including a microphone, analog-to-digital converter, digital signal processing block in an embedded system, digital-to-analog converter, and speaker.
<br>

<p>
    <img src="/assets/images/project/dsp-system-block-diagram.png"> 
    <p align="center">
    </p>
</p>

Based on the frame of hearing aid, In-the-ear(ITE), and true wireless earbuds(TWS), I set up the dynamic range of the sound based on each of the specifications for dual-microphone and speaker. After getting from the sound as scaling data in the system, overlap and add operation is used for seamless sound as natural. Based on sound generation, I contributed the application on this device as below details. 

**Applications**
- Real-time signal processing framework
- Microphone and Speaker calibration
- Sound Amplification and Compression with multi-frequency band
- Equalizer for Hearing aid and Music
- Digital Filter design to reduce the noise
- Product Verification with Python
- DSP Code Optimization to increase battery life
