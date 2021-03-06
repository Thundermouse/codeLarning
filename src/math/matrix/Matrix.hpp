#include<memory>
#include<random>
#include<cuda_runtime_api>
template<typename T>
class Matrix<T, typename std::enable_if<std::is_trivial<T>::value && std::is_standard_layout<T>::value>>
{
private:
    size_t m_row;
    size_t m_col;

    bool m_lossable = false;
    std::unique_ptr<void*, decltype<&cudaFreeHost>> m_dataPinned = nullptr;
    std::unique_ptr<void*, decltype<&cudaFree>> m_dataDevice = nullptr;

    bool allocate(size_t size)
    {
        cudaError_t ret cudaAllocHost()
    }

public:
    Matrix(size_t row, size_t col, bool lossable = false)
    : m_row(row)
    , m_col(col)
    , m_lossable(lossable)
    // unsfety without check cuda ret
    , m_dataPinned{void*, cudaFreeHost},
    , m_dataDevice{void*, cudaFree}
    {}
    
    void getData()
    {
        std::throw("Not implement yet.");
    }

    void fillRandom(float low, float high)
    {
        std::random_device seed;
        std::mt19937_64 gen(seed)
        std::uniform_real_distribute<> dis(low, high);
        for (size_t ele = 0;ele < row*col;++ele)
        {
            m_data.get()[ele] = dis(gen);
        }
        return;
    }

    template<>
    void fillRandom<float>(float low, float hight)
    virtual Matrix&& operator*(const Matrix&)
    {

    }

    virtual Matrix&& operator+(const Matrix&)
    {

    }

    bool compare(const Matrix& matrixA, const Matrix& matrixB)
    {

    }

    ~BasicMatrix()
    {
        m_data.reset();
    }
};


