---
title: Smart Pointer
aside:
    toc: true
key: 20230613
tags: C++
---

C로 프로그램을 짜다 보면 종종 실수하는 부분이 있습니다. 그건 바로 malloc과 new를 통한 동적 메모리 할당과 짝꿍인 “동적 메모리 해체”니다. 이로 인해 발생하는 메모리 누수 문제입니다. 이를 위해 **C++11 이후에 도입된 개념**이 바로 **Smart Pointer** 입니다. 한 번 예시를 봐보겠습니다.

```cpp
// From. https://www.geeksforgeeks.org/smart-pointers-cpp/

// C++ program to demonstrate the working of Template and
// overcome the issues which we are having with pointers
#include <iostream>
using namespace std;

// A generic smart pointer class
template <class T> class SmartPtr {
	T* ptr; // Actual pointer
public:
	// Constructor
	explicit SmartPtr(T* p = NULL) { ptr = p; }

	// Destructor
	~SmartPtr() { delete (ptr); }

	// Overloading dereferencing operator
	T& operator*() { return *ptr; }

	// Overloading arrow operator so that
	// members of T can be accessed
	// like a pointer (useful if T represents
	// a class or struct or union type)
	T* operator->() { return ptr; }
};

int main()
{
	SmartPtr<int> ptr(new int());
	*ptr = 20;
	cout << *ptr;
	return 0;
}
```

스마트 포인터의 개념은 생각보다 간단합니다. 위의 예제에서 보면 클래스의 생성자(Constructor)에서 메모리할당을 한 것을 클래스의 소멸자(Deconstructor)에서 할당된 메모리의 포인터를 해제하는 것을 볼 수 있습니다. 그리고 * 로는 가리켰던 레퍼런스를 → 로 읽을 수 있습니다. 이 클래스를 포인터처럼 사용한다면, 매 번 메모리 해제를 해야하는 번거로움을 줄일 수 있을 겁니다. 그럼 언제 우리는 메모리를 해제해야 할까요?

여기서 도입한 개념이 **Reference Counting**이다. 예를 들어,

```cpp
int b = 10;
int *a;
a = &b;
```

이런 경우에 b 객체를 a가 가리키게 되는데, 이를 참조라고 하고 참조당한 객제를 레퍼런스라고 합니다. Smart Pointer는 이 참조 횟수를, Reference Counting를 세는 것인데 스마트 포인터가 가르키는 객체는 이 **Reference Counting이 0이 될 때, Smart Pointer 클래스에 할당된 메모리를 해제시킵니다**.

헷갈릴 수 있으니, 더 자세한 예시를 C++에 도입한 클래스들과 함께 소개해보겠습니다. Smart Pointer의 종류는 크게 세 가지로, **unique, shared, weak 타입**이 있다.  

## 1. Types of Smart Pointer

### 1.1 unique_ptr

Unique의 경우 단어 그대로 “독보적”입니다. 그 말인 즉슨, 참조당할 수 없다는 것을 의미한다. 만약 다른 포인터로 참조를 하고 싶다면, `std::move`를 이용하여 참조하는 포인터를 이동하면 됩니다.

```cpp
// https://www.geeksforgeeks.org/smart-pointers-cpp/

// C++ program to demonstrate the working of unique_ptr
// Here we are showing the unique_pointer is pointing to P1.
// But, then we remove P1 and assign P2 so the pointer now
// points to P2.

#include <iostream>
using namespace std;
// Dynamic Memory management library
#include <memory>

class Rectangle {
	int length;
	int breadth;

public:
	Rectangle(int l, int b)
	{
		length = l;
		breadth = b;
	}

	int area() { return length * breadth; }
};

int main()
{
// --\/ Smart Pointer
	unique_ptr<Rectangle> P1(new Rectangle(10, 5));
	cout << P1->area() << endl; // This'll print 50

	// unique_ptr<Rectangle> P2(P1);
	unique_ptr<Rectangle> P2;
	P2 = std::move(P1);

	// This'll print 50
	cout << P2->area() << endl;

	// cout<<P1->area()<<endl;
	return 0;
}
```

### 1.2 shared_ptr

스마트 포인터의 기본형으로 shared 포인터는 한 객체가 이리저리 참조가 가능합니다. 

```cpp
// https://www.geeksforgeeks.org/smart-pointers-cpp/

// C++ program to demonstrate the working of shared_ptr
// Here both smart pointer P1 and P2 are pointing to the
// object Addition to which they both maintain a reference
// of the object
#include <iostream>
using namespace std;
// Dynamic Memory management library
#include <memory>
 
class Rectangle {
    int length;
    int breadth;
 
public:
    Rectangle(int l, int b)
    {
        length = l;
        breadth = b;
    }
 
    int area() { return length * breadth; }
};
 
int main()
{
    //---\/ Smart Pointer
    shared_ptr<Rectangle> P1(new Rectangle(10, 5));
    // This'll print 50
    cout << P1->area() << endl;
 
    shared_ptr<Rectangle> P2;
    P2 = P1;
 
    // This'll print 50
    cout << P2->area() << endl;
 
    // This'll now not give an error,
    cout << P1->area() << endl;
 
    // This'll also print 50 now
    // This'll print 2 as Reference Counter is 2
    cout << P1.use_count() << endl;
    return 0;
}
```

하지만 이런 경우, 문제점은 만약 두 객체가 서로가 레퍼런스가 되는 경우, 즉 순환참조시 동적 메모리가 해제되지 않고 메모리 누수 문제가 생깁니다. 순환참조는 실제로 트리형태의 데이터 구조인 경우 부모, 자식간에 일어날 수 있겠네요.

### 1.3 weak_ptr

바로 순환참조를 해결하기 위해 나타난 것이 weak 타입입니다. 이 `weak_ptr`로 선언된 스마트 포인터의 경우 참조가 되더라도 reference count가 올라가지 않습니다. 이는 weak pointer 클래스가 reference count를 세면서 동시에 weak reference count를 한 가지 더 하는 방식이기 때문입니다. 

- 참조시 reference count와 weak count로 나눔
- 참조해도 reference count가 안올라간다
- 이 포인터에 접근시 shared로 해서 작업해야 하고, 이 경우 lock() 을 이용한다. 자세한 내용은 이 [링크](https://modoocode.com/252)에 weak_ptr 부분을 참고 바란다.

여기서 한 가지 더 짚고 넘어갈 부분이 있습니다. 한 번은 면접에서 “순환 참조가 되는 경우는 사실상 많이 없을 텐데, 왜 weak pointer 타입을 사용할까요?” 라고 질문을 던지셨습니다. 기존에 순환참조를 해결하려고 만든 weak pointer의 원인이 실제 설계에서는 많이 보이진 않는다는 말에 “그러게… 그럼 왜 만들었을까?”라고 생각해봤습니다. 앞서서 **lock()**을 이용해서 weak_ptr를 이용하는 경우 그 포인터가 소멸돼었는지 아닌지를 확인할 수 있는데, 이 말은 포인터가 가르키는 객체가 존재하는지 아닌지를 알 수 있는 거죠. 즉, “[Dangling Pointer](https://en.wikipedia.org/wiki/Dangling_pointer)”로 알려진 이 부분을 weak pointer로 확인할 수 있는 것입니다. 

- Dangling pointer, ["to find if they happen to stay in memory.”](https://stackoverflow.com/questions/12030650/when-is-stdweak-ptr-useful)

- 예제 코드
    ```cpp
    #include <iostream>
    #include <memory>
    
    int main()
    {
    	// OLD, problem with dangling pointer
    	// PROBLEM: ref will point to undefined data!
    
    	int* ptr = new int(10);
    	int* ref = ptr;
    	delete ptr;
    
    	// NEW
    	// SOLUTION: check expired() or lock() to determine if pointer is valid
    
    	// empty definition
    	std::shared_ptr<int> sptr;
    
    	// takes ownership of pointer
    	sptr.reset(new int);
    	*sptr = 10;
    
    	// get pointer to data without taking ownership
    	std::weak_ptr<int> weak1 = sptr;
    
    	// deletes managed object, acquires new pointer
    	sptr.reset(new int);
    	*sptr = 5;
    
    	// get pointer to new data without taking ownership
    	std::weak_ptr<int> weak2 = sptr;
    
    	// weak1 is expired!
    	if(auto tmp = weak1.lock())
    		std::cout << "weak1 value is " << *tmp << '\n';
    	else
    		std::cout << "weak1 is expired\n";
    	
    	// weak2 points to new data (5)
    	if(auto tmp = weak2.lock())
    		std::cout << "weak2 value is " << *tmp << '\n';
    	else
    		std::cout << "weak2 is expired\n";
    }
    
    ```
    
## 2. How to use?

그럼 어떻게 Smart Point를 사용해야 할까요? 주로 shared pointer를 만드는 경우 `make_shared`를 사용한다. 스마트 포인터를 선언하는 형식도 가능하지만, 어쨋던 선언하고, 동적메모리도 할당해주는 두 가지 작업을 다 해야하기 때문에 그 두 가지를 포함하는 make_shared를 사용합니다.

그러면 클래스에서 자기자신을 참조하는 것은 구조체가 스마트 포인터로 선언될 경우 순환 참조가 될텐데 이는 어떻게 해야 할까요? 이 경우 `std::enable_shared_from_this`를 상속의 형태로 선언해주면 `shared_from_this`를 통해 자기자신 참조가 가능합니다. 이는 `weak_ptr`의 컨셉을 포함하고 있는데요, 예시는 아래를 참고하시면 되겠습니다(실제로 어떻게 나오는 지는 사이트에 들어가보시는 걸 추천).

```cpp
// https://en.cppreference.com/w/cpp/memory/shared_ptr

#include <iostream>
#include <memory>
 
struct MyObj
{
    MyObj() { std::cout << "MyObj constructed\n"; }
 
    ~MyObj() { std::cout << "MyObj destructed\n"; }
};
 
struct Container : std::enable_shared_from_this<Container> // note: public inheritance
{
    std::shared_ptr<MyObj> memberObj;
 
    void CreateMember() { memberObj = std::make_shared<MyObj>(); }
 
    std::shared_ptr<MyObj> GetAsMyObj()
    {
        // Use an alias shared ptr for member
        return std::shared_ptr<MyObj>(shared_from_this(), memberObj.get());
    }
};
 
#define COUT(str) std::cout << '\n' << str << '\n'
 
#define DEMO(...) std::cout << #__VA_ARGS__ << " = " << __VA_ARGS__ << '\n'
 
int main()
{
    COUT( "Creating shared container" );
    std::shared_ptr<Container> cont = std::make_shared<Container>();
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
 
    COUT( "Creating member" );
    cont->CreateMember();
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
 
    COUT( "Creating another shared container" );
    std::shared_ptr<Container> cont2 = cont;
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
    DEMO( cont2.use_count() );
    DEMO( cont2->memberObj.use_count() );
 
    COUT( "GetAsMyObj" );
    std::shared_ptr<MyObj> myobj1 = cont->GetAsMyObj();
    DEMO( myobj1.use_count() );
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
    DEMO( cont2.use_count() );
    DEMO( cont2->memberObj.use_count() );
 
    COUT( "Copying alias obj" );
    std::shared_ptr<MyObj> myobj2 = myobj1;
    DEMO( myobj1.use_count() );
    DEMO( myobj2.use_count() );
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
    DEMO( cont2.use_count() );
    DEMO( cont2->memberObj.use_count() );
 
    COUT( "Resetting cont2" );
    cont2.reset();
    DEMO( myobj1.use_count() );
    DEMO( myobj2.use_count() );
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
 
    COUT( "Resetting myobj2" );
    myobj2.reset();
    DEMO( myobj1.use_count() );
    DEMO( cont.use_count() );
    DEMO( cont->memberObj.use_count() );
 
    COUT( "Resetting cont" );
    cont.reset();
    DEMO( myobj1.use_count() );
    DEMO( cont.use_count() );
}
```

## 3. Operation for Smart Pointer
외에 아래에 여러가지 operation이 있으니 이건 그 때 그때 찾아보는 걸로!

- reset
- swap
- get
- * →
- [ ]
- use_count : return reference count
- unique
- bool
- owner_before
- func
- make_shared
- allocate_shared
- static_pointer_cast
- dynamic_pointer_cast
- const_pointer_cast
- reinterpret_pointer_cast
- get_deleter
- std::enable_shared_from_this
    - class Inheritance + week ptr concept

## 4. Reference

- [https://www.geeksforgeeks.org/smart-pointers-cpp/](https://www.geeksforgeeks.org/smart-pointers-cpp/)
- [https://modoocode.com/252](https://modoocode.com/252)
- [https://en.cppreference.com/w/cpp/memory/shared_ptr](https://en.cppreference.com/w/cpp/memory/shared_ptr)
- std::move: [https://modoocode.com/301](https://modoocode.com/301)