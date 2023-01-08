---
title: Elgenvalue and Eigenvector
aside:
    toc: true
key: 20230106
tags: LinearAlgebra
---
$$
    Av = \lambda v
$$

$$
    \begin{bmatrix}
    a_{11} & \cdots & a_{1n} \\
    \vdots & \ddots & \vdots \\
    a_{n1} & \cdots & a_{nn}
    \end{bmatrix}\begin{bmatrix}
    v_1 \\
    \vdots\\
    v_n
    \end{bmatrix} = \lambda \begin{bmatrix}
    v_1 \\
    \vdots\\
    v_n
    \end{bmatrix}
$$

1. Matrix A has eigenvector $v$ and eigenvalue $\lambda$. The number vector can be none or n as the maximum in N by N matrix.
2. Geometrically, the eigenvector for matrix A in linear transformation can preserve its direction and only change scale. For example, in 3D rotation, such as the concept for the earth, the unchangeable eigenvector for its rotation transformation is the vector for an axis of rotation, and the eigenvalue is 1.

## Eigen-decomposition

Eigenvalue and Eigenvector is closely related with square matrix diagonalization. This can be only in square matrix. First of all, considering 3x3 matrix as below,

$$
    \begin{bmatrix}
    v_{11} & v_{12} & v_{13} \\
    v_{21} & v_{22} & v_{23} \\
    v_{31} & v_{32} & v_{33}
    \end{bmatrix} \begin{bmatrix}
    \lambda_1 & 0 & 0 \\
    0 & \lambda_2 & 0 \\
    0 & 0 & \lambda_3
    \end{bmatrix} = \begin{bmatrix}
    \lambda_1 v_{11} & \lambda_2 v_{12} & \lambda_3 v_{13} \\
    \lambda_1 v_{21} & \lambda_2 v_{22} & \lambda_3v_{23} \\
    \lambda_1 v_{31} & \lambda_2 v_{32} & \lambda_3v_{33}
    \end{bmatrix}
$$

In this case, we can get three eigen vectors and eigen values.

As a generalization, we assumes Matrix A’s eigenvector $v_i$ and eigenvalue $\lambda_i$, $i=1,2,\cdots, n$

$$
    Av_i=\lambda_iv_i\text{ for }i= 1,2,3
$$

$$
    Av_1 = \lambda_1 v_1 \\
    Av_2 = \lambda_1 v_2 \\
    \ \ \ \ \ \ \ \  \vdots  \\
    Av_n = \lambda_n v_n
$$

And summarize,

$$
    A \begin{bmatrix}
    v_{1} & v_{2} & \cdots  v_{n} 
    \end{bmatrix} = 
    \begin{bmatrix}
    \lambda_1 v_{1} &\lambda_2 v_{2} & \cdots \lambda_n v_{n} 
    \end{bmatrix} =
    \begin{bmatrix}
    v_{1} & v_{2} & \cdots  v_{n} 
    \end{bmatrix}
    \begin{bmatrix}
    \lambda_1 & & & 0 \\
    & \lambda_2 & &\\
    & & \ddots & \\
    0 & & & \lambda_n 
    \end{bmatrix}
$$

According to the summary, if we assume the matrix P is eigenvectors as column vectors in Matrix A, and $\varLambda$ is the matrix that has eigenvalue as elements in diagonal matrix eigenvalue, then we can get an equation as below,

$$
    AP = P\varLambda \\
    A = P\varLambda P^{-1}
$$

Like this, Matrix A can have a decomposition with the multiplication between the matrix P having eigenvector as column vectors in Matrix A and the matrix $\varLambda$ having eigenvalue as elements in diagonal matrix eigenvalue. This is called the **Eigen-decomposition.**

If we know the Eigen-decomposition of matrix A, then we can easily compute the square of matrix A, the summation of diagonal elements in matrix A, and the polynomial of a matrix.

**$\checkmark$ determinant**

$$
    \begin{aligned} 
    det(A) &= det(P\varLambda P^{-1})\\
    &= det(P)det(\varLambda)det(P)^{-1}\\
    &= det(\varLambda) \\
    &= \lambda_1\lambda_2\dots\lambda_n
    \end{aligned}
$$

 

**$\checkmark$ square**
    
$$
    \begin{aligned}
    A^k &= (P\varLambda P^{-1})^k\\
    &= (P\varLambda P^{-1})(P\varLambda P^{-1})\cdots(P\varLambda P^{-1})\\
    &= P\varLambda^k P^{-1} \\
    &= Pdiag(\lambda_1^k,\cdots,\lambda_n^k) P^{-1}
    \end{aligned}
$$
    
**$\checkmark$ the summation of diagonal elements, trace**
    
$$
    \begin{aligned}
    tr(A) &= tr(P\varLambda P^{-1})\\
    &= tr(\varLambda) \\
    &= \lambda_1+\lambda_2+\dots+\lambda_n
    \end{aligned}
$$
    
**$\checkmark$ polynomial of a matrix**

$$
    \begin{aligned}
    f(A)&=a_0E+a_1A+\cdots+a_nA^n (f(x)=a_0+a_1x+\cdots+a_nx^n) \\
    &= a_0PP^{-1}+a_1P\varLambda P^{-1}+\cdots+a_nP\varLambda^n P^{-1}  \\
    &= P(a_0+a_1\varLambda +\cdots+a_n\varLambda^n )P^{-1}\\
    &= Pdiag(f(\lambda_1^k),\cdots,f(\lambda_n^k)) P^{-1}
    \end{aligned}
$$
    

## The condition of Eigen-decomposition

**The condition that we can have the Eigen-decomposition is matrix A ($\in \mathbb{R}^{n\times n}$)** should have linearly independent eigenvectors. **Linearly independent vector** defines that when vectors are in a set $\{v_1, \cdots, v_n\}$, these vectors cannot represent linear combinations with other vectors. **Linear combinations of vector** means  $a_1v_1 + a_2v_2 + ... + a_nv_n ,\ a_i\text{ is constant}$.

For example, as unit vectors for the coordinate axis in the space of 3-dimensions, we assume $v_1 = (1,0,0),\ v_2=(0,1,0), v_3=(0,0,1)$. We can easily confirm that even if multiplying any constant to $v_2$ and $v_3$, it must not be $v_1$. If we add $v_4=(-1, 3, 4)$ to a set of vectors above, then set $\{v_1, v_2, v_3,v_4\}$ is not linearly independent($∵ v_4 = -v_1+3v_2+4v_3$). It means three-dimension space can have three linearly independent vectors to the maximum. As a generalization, These $n$ linearly independent vectors can be basis role in $n$-dimension space to the maximum. Basis defines that the linear combination in some vectors of linearly independent vectors in a set $\{v_1,\cdots, v_n\}$ can represent all of the vectors in $n$-dimension space.

Then how can we get the eigenvector and eigenvalue? The equation representing the definition of eigenvector and eigenvalue is as below, 

$$
    \begin{aligned}
    Av &= \lambda v \\
    Av-\lambda v &= 0 \text{ (0: zero matrix, null matrix)} \\
    (A-\lambda E)v &= 0 \text{ (E: unit matrix)}
    \end{aligned}
$$

The eigenvector and eigenvalue are the solutions of the above equation(and we define $v\not=0$). Then it means if $(A-\lambda E)$ is an invertible matrix, then $v$ always exists. However, an eigenvector can exist when $(A-\lambda E)$ is not an invertible matrix because of its definition($v\not=0$). 

Therefore, we can get the condition of getting the eigenvector as “$det(A-\lambda E)= 0$”. We called this equation the **characteristic equation for matrix A**.

**$\checkmark$ Example**

$$
    A = \begin{bmatrix}
    2 & 0 & -2 \\
    1 & 1 & -2 \\
    0 & 0 & 1
    \end{bmatrix} \rightarrow (2-\lambda)(1-\lambda)^2 =0
$$

$$
    \lambda=2,\ v_x=v_y,\ v_z=0 \rightarrow v=\begin{bmatrix}
    1 & 1 & 0 
    \end{bmatrix}^T
$$

$$
    \lambda=1,\ v_x=2v_z
$$

$$
    v = (2t, s, t)
$$

When $\lambda$  is 1, the vector can represent $(2t,s,t)$ if $t$ and $s$ is a random real number, and the linear combination $(2t, s, t) = t(2, 0, 1) + s(0, 1, 0)$. This means the eigenvector can be $(2,0,1)$ and $(0,1,0)$. 

As we can find above example, the eigenvector looks ambiguous because a matrix can determine eigen value, but an eigenvector cannot determine as a specific one. 

$$
    Av = \lambda v \\
    A(cv) = \lambda (cv) 
$$

As mentioned above, if vector $v$ is an eigenvector for eigenvalue $\lambda$, then $cv$ can be an eigenvector about random constant $c$ (except zero). Likewise, if vector $v_1, v_2$ are eigenvectors for eigenvalue $\lambda$, about random constant $c_1, c_2$ (except zero),

$$
    A(c_1v_1+c_2v_2) = \lambda c_1v_1 + \lambda c_2v_2 = \lambda(c_1v_1 +c_2v_2) 
$$

$c_1v_1+c_2v_2$ also can be elgenvectors for eigenvalue $\lambda$. Therefore, generally, we define the unit vector normalizing to scale one as an eigenvector. (But, in example($\lambda=1$), it should have two eigenvectors for representing all of the vectors in space because it has two degrees of freedom)

## Symmetric Matrix's Eigen-decomposition

Symmetric matrix defines $A^T=A$ (all $i,j$ are $a_{ij}=a_{ji}$).\

If its elements are all in **real number**, this **Symmetric matrix can have Eigen-decomposition**, and even **we can use an orthogonal matrix** in Eigen-decomposition as below([Proof](http://www.quandt.com/papers/basicmatrixtheorems.pdf)),

$$
    \begin{aligned}
    A &= P\varLambda P^{-1} \\
    &= P\varLambda P^{T}, \ \text{assume } PP^T = E\ (P^{-1}=P^T) 
    \end{aligned}
$$

This property always should be in real numbers. If elements in the matrix are in complex numbers, then the unitary matrix can have Eigen-decomposition. 

The Eigen-decomposition of a symmetric matrix is a basic property for SVD(**[Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)**) and PCA(**[Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)**)

## Reference

- [https://angeloyeo.github.io/2019/07/17/eigen_vector.html](https://angeloyeo.github.io/2019/07/17/eigen_vector.html)
- 왜 공분산행렬의 고유벡터가 데이터 분포의 분산 방향이 되고, 고유값이 그 분산의 크기가 되는 것일까? [https://www.stat.cmu.edu/~cshalizi/350/lectures/10/lecture-10.pdf](https://www.stat.cmu.edu/~cshalizi/350/lectures/10/lecture-10.pdf)
- [https://darkpgmr.tistory.com/105](https://darkpgmr.tistory.com/105)
- [https://velog.io/@lorenzo/Eigendecomposition](https://velog.io/@lorenzo/Eigendecomposition)
- [https://www.youtube.com/watch?v=PFDu9oVAE-g&t=2s](https://www.youtube.com/watch?v=PFDu9oVAE-g&t=2s)