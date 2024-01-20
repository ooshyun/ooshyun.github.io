---
title: Basic concept for Machine learning using statistics
aside:
    toc: true
key: 20230124
tags: ML
---
### What is p-value ?

$$
p = P(\text{Assumption} \lvert \text{ Observation})\ ?
$$

p-value, "Null hypothesis"

$$
    \text{p value }=\text{ 1 case + 2 case + 3 case}
$$

1. The probability random chance would result in the observation
2. The probability of observing something else tha is equally rare
3. The probablity of observing something rarer or more extreme

- Example for p-value(Two-Sided)
    
    1. [Null hypothesis, opposite meaning is "My coin is super special because it landed on Heads twice in a row] In statistic lingo, the hypothesis is even though I got HH in a row, my coin is no different from a normal coin. (H: Head, T: Tail) In this case, the probability is HH 0.25, TT 0.25, HT/TH 0.5. It didn't matter a order because the first flip didn't effect the next flip.

    $$
        \text{p-value for 2 Heads }= 0.25 + 0.25 + 0 = 0.5
    $$

    2. [Null hypothesis] In statistic lingo, the hypothesis is even though I got HHHTH in a row, my coin is no different from a normal coin. (H: Head, T: Tail)
    
    $$
        \text{p-value for HHHTH }= \dfrac{5}{32} + \dfrac{5}{32} + \dfrac{2}{32} = 0.375
    $$

    - Probability distribution (refer to the [youtube](https://www.youtube.com/watch?v=JQc3yx0-Q9E&t=702s)!)

- One-sided p-value 

### Linear regression

$$
\hat\beta^{LS} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 \} = (X^TX)^{-1}X^Ty
$$

$$
\begin{aligned}

E(x)&=\displaystyle\sum_{i=1}^Nx_iP(x_i) \\

V(x)&= E((x-m)^2)=\dfrac{1}{N}\displaystyle\sum_{i=1}^N(x_i-m)^2 \\
&= E((x-E(x))^2) = \displaystyle\sum_{i=1}^N P(x_i)(x_i-E(x))

\end{aligned}

$$

$$
\begin{aligned}
\text{Expected MSE} &= E[(Y-\hat{Y})^2\lvert X] \\
&= \sigma^2 + (E[\hat Y] -\hat Y)^2 + E[\hat Y -E[\hat Y]]^2 \\
&= \sigma^2 -\text{Bias}^2(\hat Y) + Var(\hat Y) \\
&= \text{Irreducible Error} + \text{Bias}^2 + \text{Variance}
\end{aligned}
$$

**$\checkmark$ The goal is removing most of $\beta$, and simplify the linear model!**

### Ridge regression, “L2”

$$
\hat\beta^{ridge} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 \}, \text{ subject to }\displaystyle\sum_{j=1}^p \beta_j^2 \leq t
$$

Represent using **[Lagrange multiplier,](https://www.notion.so/Statistic-Lagrange-multiplier-66de3a1fff994b7795d30fd3a89e2b84)**

$$
\hat\beta^{ridge} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 + \lambda \displaystyle\sum_{j=1}^p \beta_j^2 \}
$$

<p>
    <img src="/assets/images/post/machinelearning/basic-concept/Ridge.png" width="200" height="400" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption">Youtube, [핵심 머신러닝] 정규화모델 1(Regularization 개념, Ridge Regression)</em>
    </p>
</p>


$\checkmark$ **Cons: Almost $\beta_2$ can be 0, but not 0.** 

$\checkmark$ **Ridge regression is differentiable → closed form solution**

### Lasso(Least Absolute Shrinkage and Selection Operator), “**L1”**

$$
\hat\beta^{ridge} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 \}, \text{ subject to }\displaystyle\sum_{j=1}^p \beta_j \leq t
$$

$$
\hat\beta^{ridge} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 + \lambda  \displaystyle\sum_{j=1}^p \beta_j \}
$$

<p>
    <img src="/assets/images/post/machinelearning/basic-concept/Lasso.png" width="200" height="400" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption">Youtube, [핵심 머신러닝] 정규화모델 1(Regularization 개념, Ridge Regression)</em>
    </p>
</p>

$\checkmark$ **Cons:  $\beta_2$ can be 0, but if data has high covariance, then it lose its robustness**

### Elastic net, “**L1+L2”**

$$
\hat\beta^{\text{Elastic net}} = \underset{argmin}{\beta}\{     \displaystyle\sum_{i=1}^n (y_i-x_i\beta)^2 + \lambda_1\displaystyle\sum_{j=1}^p \beta_j^2 + \lambda_2\displaystyle\sum_{j=1}^p \beta_j \} \}
$$

<p>
    <img src="/assets/images/post/machinelearning/basic-concept/ElasticNet.png" width="400" height="300" class="projects__article__img__center">
    <p align="center">
    <em class="projects__img__caption">Figure: An image visualising how ordinary regression compares to the Lasso, the Ridge and the Elastic Net Regressors. Image Citation: Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. </em>
    </p>
</p>

### Reference
- [p-values: What they are and how to interpret them](https://www.youtube.com/watch?v=vemZtEM63GY)
- [How to calculate p-values](https://www.youtube.com/watch?v=JQc3yx0-Q9E)
- [Youtube, [핵심 머신러닝] 정규화모델 1(Regularization 개념, Ridge Regression)](https://www.youtube.com/watch?v=pJCcGK5omhE&t=21s)