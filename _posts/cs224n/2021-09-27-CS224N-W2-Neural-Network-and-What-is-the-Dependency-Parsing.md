---
title: CS224N W2. Neural Network and What is the Dependency Parsing
sidebar:
    nav: cs224n-eng
aside:
    toc: true
key: 20210713
tags: CS224N
---
**All contents is arranged from [CS224N](https://online.stanford.edu/artificial-intelligence/free-content?category=All&course=6097) contents. Please see the details to the [CS224N](https://online.stanford.edu/artificial-intelligence/free-content?category=All&course=6097)!**

## 1. Named Entity Recognition(NER)

1. Task: **Find and Classify** names in text

2. Example 
    <p>
        <img src="/assets/images/post/cs224n/w2/ner-example.png" width="400" height="100" class="projects__article__img__center">
        <p align="center">
        <em class="projects__img__caption"> Reference. Stanford CS224n, 2021</em>
        </p>
    </p> 

3. Usages
    - Tracking mentions of particular entities in documents
    - For question answering, answers are usually named entities
    - Often followed by Named Entity Linking/Canonicalization into Knowledge Base

4. Simple NER: Window Classification using ninary logistic classifier
    
    - Idea: Classify each word in the context window of neighboring words
    
    - Train logistic classifier on hand-labeled data to classify center word(yes/no) for each class based on a concatenation of word vectors in a window
    
    - Example: Classify "Paris" as $$+$$ or $$-$$ location in context of sentence with window length 2

        <p>
            <img src="/assets/images/post/cs224n/w2/simple-ner-example.png" width="150" height="50" class="projects__article__img__center">
            <p align="center">
            <em class="projects__img__caption"> Reference. Stanford CS224n, 2021</em>
            </p>
        </p> 

    - Resulting $$x_{window} = x \in R^{5d}$$, a column vector

    - To classify all words, run classifier for each class on the vector cnetered on each word in the sentence

5. Binary Classification for center word being location

    - Model
        <p>
            <img src="/assets/images/post/cs224n/w2/ner-binary-classification.png" width="300" height="300" class="projects__article__img__center">
            <p align="center">
            <em class="projects__img__caption"> Reference. Stanford CS224n, 2021</em>
            </p>
        </p> 

    - Equation

        $$
            s = u^Th, h=f(Wx+b), x(input) \\
            J_t(\theta) = \sigma(s)=\dfrac{1}{1+e^{-s}}
        $$

## 2. Stochastic Gradient Descent in Neural Network 

**$\checkmark$  [Mathmatic for Stochastic Gradient Descent in Neural networks](/2021/07/13/stochastic-gradient-descent-nn)**

## 3. Neural Network

**$\checkmark$  [The Concept of Neural Network and Technique](/2021/07/13/neural-network)**

## 4. Dependency Parsing
**$\checkmark$ Constituency Grammar**
- Two views of linguistic structure: Constituency

        = phrase structure grammar 
        = context-free grammars (CFGs)
    
- Phrase structure organizes words into nested constituent
    1. Starting unit: words(ex. the, cat, cuddly, by, door)
    2. Words combine into phrases(ex. the cuddly cat, by the door)
    3. Phrases can combine into bigger phrases(ex. the cuddly cat by the door)

**$\checkmark$ [Dependency Parsing](/2021/07/13/dependency-parsing)**