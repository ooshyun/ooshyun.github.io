---
title: Taylor Series Approximation and Error
aside:
    toc: true
key: 20230702
tags: Math
---
## Prologue

모델 경량화 기법중에 Pruning의 하나에서 [Taylor Expansion Analysis on Pruning Error](https://www.notion.so/MIT-6-S965-Fall-2022-TinyML-and-Efficient-Deep-Learning-Computing-bdbe14872a4e4648941d2c4cfeb4e798?pvs=21)에서 최소값을 구한다고 나온 증명속 테일러 급수… 넌 누구니…?

<!--more-->

$$
\delta L = L(x;W)-L(x;W_p=W-\delta W) \\ = \sum_i g_i\delta w_i + \frac{1}{2} \sum_i h_{ii}\delta w_i^2 + \frac{1}{2}\sum_{i\not=j}h_{ij}\delta w_i \delta w_j + O(\lvert\lvert \delta W \lvert\lvert^3) \\

where\ g_i=\dfrac{\delta L}{\delta w_i}, h_{i, j} = \dfrac{\delta^2 L}{\delta w_i \delta w_j}
$$

## 1. Taylor Series

Tayler Series 는 임의의 함수 $f(x)$를 polynomial로  아래와 같이 표현했을 때 아래와 같습니다.

$$
f(x) = a_0 + a_1x + a_2x^2 + a_3x^3 + \cdots
$$

Tayler의 목표는 식을 간단하게 표현하는 것입니다. 간단하게 표현하기 위해 우선 $x_0=0$ 이라 가정하고 다차 미분식을 구해 보면 아래와 같습니다. 

<br>

$$
\begin{aligned}
&f'(x) = a_1 + 2a_2x + 3a_3x^2 + 4a_4x^3 + \cdots \\
&f''(x) = 2a_2 + (3\times 2)a_3x + (4\times 3)a_4 x^2\cdots \\
&f'''(x) = (3\times 2)a_3 + (4\times 3\times 2)a_4 x + \cdots \\
&f'^{v}(x) = (4\times 3\times 2)a_4 + \cdots \\
\end{aligned}
$$

$$
\begin{aligned}
&f(0) = a_0,\ f'(0)=a_1, \\
&f''(0) = 2a_2,\ f'''(0) = (3\times 2) a_3, \\
&f'^v(0)=(4\times 3\times 2) a_4
\end{aligned}
$$

<br>

위의 식을 이용하면 $x_0=0$ 일때 Tayler Series Approximation 은 아래와 같을 수 있습니다

<br>

$$
\begin{aligned}
&f(x)= a_0 + a_1x + a_2x^2 + a_3x^3 + \cdots \\
&f(x) = \sum_{i=0}^\infty a_ix^i = \sum_{i=0}\dfrac{f^{i}}{i!}x_i
\end{aligned}
$$

<br>

조금 더 일반화된 식으로 표현해보면 **$x_0$ 에서 Tayler Series Approximation**은 바로 아래 식입니다.

<br>

$$
\begin{aligned}
&f(x) = f(x_0) + f'(x_0)(x-x_0) + \dfrac{f''(x_0)}{2}(x-x_0)^2 + \dfrac{f'''(x_0)}{3}(x-x_0)^3 + \cdots \\
&f(x) = \sum_{i=0}^{\infty}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i
\end{aligned}
$$

<br>

그런데 항상 저희는 특정 차수까지 “Truncate”해서 식을 사용할 수 밖에 없습니다. 그럼 만약 Truncation을 했을 때 에러는 얼마로 추정할 수 있을까요?

## 2. Remainder of Talyer Series

$h=x-x_0$라고 가정하고 식을 정리하면 다음과 같이 나눌 수 있습니다.

<br>

$$
\begin{aligned}
&f(x_0+h) = f(x_0) + f'(x_0)h + \dfrac{f''(x_0)}{2}h^2 + \dfrac{f'''(x_0)}{3}h^3 + \cdots \\
&f(x_0+h) = 
\sum_{i=0}^{n}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i +
\sum_{i=n+1}^{\infty}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i
\end{aligned}
$$

<br>

여기서 $\sum_{i=0}^{n}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i$를 truncated 부분이라고, 즉 n차 Talyor approximation라고 하면 error는 $\sum_{i=n+1}^{\infty}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i$이 됩니다. 이제 error를 추정해볼까요?

<br>

$f$가 $(x_0, x)$에서  $(n+1)$번 미분가능하고 $f^{(n)}$이 $[x_0, x]$에서 연속이라고 가정해 봅시다. 그러면 식은 아래와 같이 정리할 수 있습니다.

<br>

$$
\begin{aligned}
error = \text{exact - approximation} &= f(x) - t_n(x) = \sum_{i=n+1}^{\infty}\dfrac{f^{(i)}(x_0)}{i!}(x-x_0)^i \\
&= \dfrac{f^{(n+1)}(x_0)h^{n+1}}{(n+1)!}h^{n+1} + \dfrac{f^{(n+2)}(x_0)h^{n+2}}{(n+2)!}h^{n+2} + \cdots
\end{aligned}
$$

<br>

**여.기.서. $\dfrac{f^{(n+1)}(x_0)h^{n+1}}{(n+1)!}h^{n+1}$ 는 $h \rightarrow 0, x \rightarrow x_0$ 으로 갈 때 식에서 dominant해질 수 있습니다(h가 1보다 작기 때문이지요). 그럼 위 식은 아래처럼 간략화해 생각할 수 있겠습니다.**

<br>

$$
error \leq Mh^{n+1}\ or\ error = O(h^{n+1})
$$

<br>

마지막 Term에서 오늘의 질문의 해답이 보입니다. 정리는 마지막에 가서 해보겠습니다. 그럼 Remainer를 보기 좋게 정리해보겠습니다. $h=x-x_0$이고  $f$가 $(x_0, x)$에서  $(n+1)$번 미분가능하고 $f^{(n)}$이 $[x_0, x]$에서 연속이라고 가정하면,

<br>

$$
error = \text{exact - approximation}\\

\begin{aligned}
Remainder\ Theorem: R_n &= f(x)-t_n(x) = \sum_{i=n+1}^{\infty}\dfrac{f^{(i)}(x_0)}{i!}h^i \\

&= \dfrac{f^{(n+1)}(\delta)}{(n+1)!} (\delta-x_0)^{n+1}, where\ \delta \in (x_0,x)

\end{aligned}
$$

<p>
    <img src="/assets/images/post/math/taylor-series/equation.png" width="500" height="100" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption">Reference. Lecture 8 in [https://courses.engr.illinois.edu/cs357/fa2019]</em>
    </p>
</p>

<br>

근데, 이 식은 그래서 어떻게 이용해야 할까요?

## 3. Example

만약 아래와 같은 그래프에서 $error = f(x)-t_1(x)$ 가 되겠습니다. 그리고 그 $t_1(x)$을 추정하기 위해서 우리는 임의의 가까운 지점 $x_0$에서 접선을 그었을 때 $(x, t_1(x))$를 표현할 수 있고, 이는 error를 표현할 수 있습니다.

<p>
    <img src="/assets/images/post/math/taylor-series/example.png" width="500" height="400" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption">Reference. Lecture 8 in [https://courses.engr.illinois.edu/cs357/fa2019]</em>
    </p>
</p>

만약 앞선 Tayler Series Approximation에 따르면 지금은 1차 함수로 Approximation을 하고 있으니 $error=O(h^2)$이 되겠군요. 이 말은 $h=x-x_o$ 이므로 $x$ 와 $x_0$의 거리가 가까워 지면 $error$ 는 제곱만큼 작아진다는 말이 됩니다.

## Epliloge

음, 그럼 다시 처음 질문으로 돌아가 보겠습니다.

$$
\delta L = L(x;W)-L(x;W_p=W-\delta W) \\ = \sum_i g_i\delta w_i + \frac{1}{2} \sum_i h_{ii}\delta w_i^2 + \frac{1}{2}\sum_{i\not=j}h_{ij}\delta w_i \delta w_j + O(\lvert\lvert \delta W \lvert\lvert^3) \\

where\ g_i=\dfrac{\delta L}{\delta w_i}, h_{i, j} = \dfrac{\delta^2 L}{\delta w_i \delta w_j}
$$

모델 경량화에 자세한 내용을 다루지 않겠지만, 위 식은 $\delta L$ 를 최소화하는게 목적으로 보이고 식을 보니 2nd order Tayler series approximation로 추정됩니다. $O(\lvert\lvert\delta W \lvert\lvert^3)$는 Taylor Series가 second-order로 가정했을 때 3-order error라는 의미이며, $\frac{1}{2} \sum_i h_{ii}\delta w_i^2 + \frac{1}{2}\sum_{i\not=j}h_{ij}\delta w_i \delta w_j$ 는 second-derivate term 일 것입니다.

여기까지 Tayler Series Approximation과 Reminder, error에 대해서 살펴봤습니다. 더 자세한 내용은 아래 레퍼런스를 참고해주시기 바랍니다.

## Reference

- [https://courses.engr.illinois.edu/cs357/fa2019/assets/lectures/Lecture8-Sept19.pdf](https://courses.engr.illinois.edu/cs357/fa2019/assets/lectures/Lecture8-Sept19.pdf)