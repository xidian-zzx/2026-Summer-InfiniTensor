#include "../exercise.h"

// READ: 左值右值（概念）<https://learn.microsoft.com/zh-cn/cpp/c-language/l-value-and-r-value-expressions?view=msvc-170>
// READ: 左值右值（细节）<https://zh.cppreference.com/w/cpp/language/value_category>
// READ: 关于移动语义 <https://learn.microsoft.com/zh-cn/cpp/cpp/rvalue-reference-declarator-amp-amp?view=msvc-170#move-semantics>
// READ: 如果实现移动构造 <https://learn.microsoft.com/zh-cn/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170>

// READ: 移动构造函数 <https://zh.cppreference.com/w/cpp/language/move_constructor>
// READ: 移动赋值 <https://zh.cppreference.com/w/cpp/language/move_assignment>
// READ: 运算符重载 <https://zh.cppreference.com/w/cpp/language/operators>

class DynFibonacci {
    size_t *cache;
    int cached;

public:
    // TODO: 实现动态设置容量的构造器
    DynFibonacci(int capacity)
        : cache(new size_t[capacity]{0, 1}),
          cached(1) {
        ASSERT(capacity >= 2, "capacity should be at least 2");
    }

    // TODO: 实现移动构造器
    DynFibonacci(DynFibonacci &&other) noexcept
        : cache(other.cache),
          cached(other.cached) {
        // other 已经不再拥有这块内存。
        //
        // 必须将其指针设为空，否则：
        // 1. 当前对象析构时会 delete[] cache
        // 2. other 析构时也会 delete[] cache
        //
        // 这会造成同一块内存被释放两次。
        other.cache = nullptr;
        other.cached = 0;
    }

    // TODO: 实现移动赋值
    // NOTICE: ⚠ 注意移动到自身问题 ⚠
    DynFibonacci &operator=(DynFibonacci &&other) noexcept {
        // 必须检查是否为自身移动赋值：
        //
        // fib0 = std::move(fib0);
        //
        // 如果没有这个判断，可能先释放自己的 cache，
        // 然后再从已经被释放的自己那里接管指针。
        if (this != &other) {
            // 先释放当前对象原本拥有的动态数组，
            // 否则直接覆盖 cache 会造成内存泄漏。
            delete[] cache;

            // 接管 other 的资源。
            cache = other.cache;
            cached = other.cached;

            // other 放弃资源所有权，进入“已移动”状态。
            //
            // 已移动对象仍然可以安全析构，
            // 但这里不应该再用它计算斐波那契数。
            other.cache = nullptr;
            other.cached = 0;
        }
            return *this;
    }

    // TODO: 实现析构器，释放缓存空间
        ~DynFibonacci() {
        delete[] cache;
    }

    // TODO: 实现正确的缓存优化斐波那契计算
    size_t operator[](int i) {
        // 已移动对象的 cache 是 nullptr，不能继续访问。
        ASSERT(cache != nullptr, "object has been moved");

        ASSERT(i >= 0, "index should not be negative");

        // 如果目标下标还没有计算，就继续向后计算。
        while (cached < i) {
            // 先移动到下一个尚未计算的位置。
            ++cached;

            // 根据前两项计算当前项。
            cache[cached] =
                cache[cached - 1] + cache[cached - 2];
        }

        return cache[i];
    }

    // NOTICE: 不要修改这个方法
    size_t operator[](int i) const {
        ASSERT(i <= cached, "i out of range");
        return cache[i];
    }

    // NOTICE: 不要修改这个方法
    bool is_alive() const {
        return cache;
    }
};

int main(int argc, char **argv) {
    DynFibonacci fib(12);
    ASSERT(fib[10] == 55, "fibonacci(10) should be 55");

    DynFibonacci const fib_ = std::move(fib);
    ASSERT(!fib.is_alive(), "Object moved");
    ASSERT(fib_[10] == 55, "fibonacci(10) should be 55");

    DynFibonacci fib0(6);
    DynFibonacci fib1(12);

    fib0 = std::move(fib1);
    fib0 = std::move(fib0);
    ASSERT(fib0[10] == 55, "fibonacci(10) should be 55");

    return 0;
}
