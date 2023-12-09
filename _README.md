* Quick Start: 
    https://github.com/kitian616/jekyll-TeXt-theme
* Ruby/JekyII setup: 
    https://www.notion.so/ooshyun/SETUP-a8239fb0d6bb4332a542b58b91a1f515

* Generate Gemfile.lock: 
    docker run --rm -v "$PWD":/usr/src/app -w /usr/src/app ruby:2.6 bundle install
* Build Docker image: 
    docker-compose -f ./docker/docker-compose.build-image.yml build
* Run local server(http://localhost:4000/): 
    docker-compose -f ./docker/docker-compose.default.yml up

* Post Upload
    Add the post in _post directory

* Add the Category
    - Add navigation.yml
    - If want to add header such as about in category, then modify that file

* Add nav/toc(SubCategory)
    1. _data/navigation에 navigate 항목을 추가한다.
        각 페이지마다 nav를 설정해줘야 nav가 보인다!
        - 예제. file: _data/navigation.yml
            header:
            - titles:
                ko      : &KO       수식이 깨져 보일 때
                ko-KR   : *KO
                url: 2019/01/03/when_latex_not_appear.html

            - titles:
                ko      : &KO       Youtube 채널
                ko-KR   : *KO
                url: https://www.youtube.com/channel/UCoCrPRq7wa6q0p59abF5aNw?view_as=subscriber

            docs-ko:
            - title:      기초 선형대수학 이론
                children:
                - title:  벡터의 기본 연산(상수배, 덧셈)
                    url:    2020/09/07/basic_vector_operation.html
                - title:  행렬 곱에 대한 또 다른 시각
                    url:    2020/09/08/matrix_multiplication.html
                - title:  행벡터의 의미와 벡터의 내적

            - title:      선형대수학 응용
                children:
                - title:  주성분분석(PCA)
                    url:    2019/07/27/PCA.html
                - title:  고윳값 분해(EVD)
                    url:    2020/11/19/eigen_decomposition.html
            ...

    2. nav가 보여지고 싶은 페이지마다 nav를 추가해줘야 한다.
        * 날짜 기입시 주의할 것! /2022/06/18/test-for-post.html
        - Example, Reference. https://github.com/angeloyeo/angeloyeo.github.io
            - _posts/xx.md
                ---
                title: 수식이 깨져 보일 때 대처법
                sidebar:
                    nav: docs-ko
                aside:
                    toc: true
                key: 20190103
                tags: 도움말
                ---

* Change logo
    Change _include/svg/logo.svg

* Change favicon
    After getting the resource from https://realfavicongenerator.net/
        1. Change _include/head/favicon.html
        2. Change assets/*
        3. Change /favicon.ico
        4. head = tab image, header = the top of the page(such as logo, title, and navigator)
    - Reference
    1. [Quick Start](https://tianqi.name/jekyll-TeXt-theme/docs/en/quick-start)
    2. https://realfavicongenerator.net/

* Add Caption in markdown
    https://geniewishescometrue.tistory.com/entry/%EB%A7%88%ED%81%AC%EB%8B%A4%EC%9A%B4-%EA%B4%80%EB%A0%A8-%ED%8C%81-%EA%B8%80-%EC%83%89%EC%83%81-%ED%98%95%EA%B4%91%ED%8E%9C

* Issue
    - _project.md cannot build in Jeklly
    - header icon is red!!!

* Add comment
    - https://velog.io/@neori/Jekyll-TeXt-Theme로-GitHub-블로그-개발하기
    - https://yenilee.github.io/2021/07/21/configure-gitalk.html
    - github Oauth application -> _config.yml/comments
    - github Oauth application: https://github.com/settings/developers
    
        ```markdown
        Application name: ooshyun.log
        Homepage URL: https://ooshyun.github.io/
        Application description: // optional
        Authorization callback URL: https://ooshyun.github.io/
        ```
        
    - _config.yml

        ```markdown
        ## Gitalk
        # please refer to https://github.com/gitalk/gitalk for more info.
        gitalk:
            clientID    : 'e08a1e93072c38209a89' # GitHub Application Client ID
            clientSecret: 'f5dc8af9855f6eb1a8342033c58a5f5314b1cd8f' # GitHub Application Client Secret
            repository  : 'ooshyun.github.io.comments' # GitHub repo
            owner       : 'ooshyun' # GitHub repo owner
            admin:  ['ooshyun'] # GitHub repo owner and collaborators, only these guys can initialize GitHub issues, IT IS A LIST.
            # - your GitHub Id
        ```
    - comment repository should be public and opened issues in settings

- 목표
    - [V] My awesome websites 밑에 문구 넣기 + size up logo 
    - [V] about: [TODO] 소개 문구 추가 수정
        - [TODO] email, github, notion -> icon
    - [V] Projects 
        - [TODO] On-going project 표시해놓기
        - [V] Project 페이지 사진 넣어서 링크넣기
        - Project 페이지 정리하기
            - Project Page
                - [V] Olive Max and Pro, 2022
                    - [TODO] Research noise name 
                    - [TODO] 기록한 것과 Compression, Calibration 링크 이어 놓기
                - [V] Speech Enhancement using LSTM, 2022, [Code]  
                    - [TODO] Research the result picture
                    - [TODO] TSNR based noise reduction도 포함시키기
                    - [TODO] Document 정리해서 github 링크 달기
                - [V] Design Equalizer with cascasde and parallel biquid filter, Fall 2021, [Code]
                - [X] Transceiver and Receiver for single-ended PAM2 with differential sensing, Fall 2019
                - [V] Delay Locked Loop for DDR3 and LPDDR3, 2019
                - [V] PHY Interface for DDR3 and LPDDR3, 2018
        - [V] About, Projects grammerly
        - [V] Activities Page
            - [TODo] Same format as Project?
            - Academic Scholarship            
                - 뜬금 없음[TODO]
                - The entire writing is not shown as each highlight -> convert to dot or (-)?

            - Mentoring and Volunteering
                - Mentoring
                - Samsung Volunteering
                - [TODO][#0] 사진    
            - Personal Project
                - Brain based psycholic -: exp and video
                - Linguistics : exp and paper 
            - Club Activity: Pic, comment

    - [ING] Writing 정리
        - [ ]CS224N
            - [ ]WEEK 1
            - [ ]WEEK 2
            - [ ]WEEK 3
            - [ ]WEEK 4
            - [ ]WEEK 5
        - Programming/Setup
            - [ ]Git
        - Programming/Concept
            - [ ]Digitial Signal Processing
            - [ ]Engineering/Microphone Calibration
            - 여기까지가 목표!
            - [ ]Speech and Natural Language
            - [ ]Engineering Mathematics
            - [ ]Analog Circuit design
            - [ ]Python
            - [ ]Algorithm
            - [ ]Machine Learning
            - [ ]C, C++
            - [ ]Hardware
            - [ ]KATEX
            - [ ]Linear Algebra
            - [ ]ETC
            - [ ]Linux
            - [ ]Book
        - Programming/CodeBug(BugFix)
        
    - 썸네일 만들기(링크를 올리면 미리보기가 안보여!)
        - _config.yml에서 excerpt_separator 옵션
        - http://www.seanbuscay.com/blog/jekyll-teaser-pager-and-read-more/
        - 글에 <!--more--> 표시하면 됌

    - [TODO] 2023.05.06 
        - [ ] 글별 카테로리 이름에 넣기
        - [ ] 카테고리별로 글 볼 수 있도록 하기(폴더별로 보듯)
        - [ ] 댓글 달기
        - [ ] toc 블럭 키우기(글자가 잘려,,,)
        - [ ] Project에 음악 visualizer 넣기
            ```markdown
            ## Side Projects
            <div class="about__project__font">
                Jammy <br>
                <div class="duration__font">
                Spring 2023 <a href="{% link assets/music-vis/jonejkim.github.io/music-vis-d3js/index.html %}">Read more</a> <br>
                </div>
            </div>
            <audio id="audio" src="./assets/music-vis/jonejkim.github.io/music-vis-d3js/assets/kubbiJam.mp3" crossorigin="anonymous" ></audio>
            ```

    - TBD
        - Interview
            - notion: https://cochl.oopy.io/4aa042ea-8748-499c-974a-fbd630c29a9d
            - medium: https://medium.com/cochl/meet-cochls-team-seunghyun-oh-back-end-engineer-e1b3c05c39fe



- _include/head.html: the top of the sites Home - Seunghyun Oh
- _include/header.html: the top of the pages, character and name
- _sass/_header.scss: the style of contents related w/ header
- _include/home.html: when url at first, it shows
- custom style: _sass/custom.scss
- header style: _sass/components/_header.scss
- photo style: _sass/additional/_photo-frame.scss
- Reference for link in markdown and html: https://jekyllrb.com/docs/liquid/tags/#links
- How to add image in markdown: [![Text](Image URL or Path)](Link URL)

- good example
- https://www.christiansteinmetz.com/projects
- https://minalee.info/
