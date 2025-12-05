# SpeakerEmbeddingManager 并发安全性分析

## 概述

本文档详细分析了 `SpeakerEmbeddingManager` 的并发安全性，包括锁机制检查、潜在的并发问题以及使用建议。

## 执行摘要

**结论：`SpeakerEmbeddingManager` 不是线程安全的，不支持并发访问。**

- ❌ **核心 C++ 实现中没有任何锁机制**
- ❌ **所有语言绑定（Java、Kotlin、Python、Go、C#等）都没有额外的同步机制**
- ⚠️ **并发使用会导致数据竞争（Data Race）和未定义行为**

## 核心实现分析

### 1. C++ 核心实现

#### 1.1 数据结构

核心实现在 `sherpa-onnx/csrc/speaker-embedding-manager.cc` 中，`SpeakerEmbeddingManager::Impl` 类包含以下共享数据结构：

```21:232:sherpa-onnx/csrc/speaker-embedding-manager.cc
class SpeakerEmbeddingManager::Impl {
 public:
  explicit Impl(int32_t dim) : dim_(dim) {}

  bool Add(const std::string &name, const float *p) {
    if (name2row_.count(name)) {
      // a speaker with the same name already exists
      return false;
    }

    embedding_matrix_.conservativeResize(embedding_matrix_.rows() + 1, dim_);

    std::copy(p, p + dim_, &embedding_matrix_.bottomRows(1)(0, 0));

    embedding_matrix_.bottomRows(1).normalize();  // inplace

    name2row_[name] = embedding_matrix_.rows() - 1;
    row2name_[embedding_matrix_.rows() - 1] = name;

    return true;
  }

  bool Add(const std::string &name,
           const std::vector<std::vector<float>> &embedding_list) {
    if (name2row_.count(name)) {
      // a speaker with the same name already exists
      return false;
    }

    if (embedding_list.empty()) {
      SHERPA_ONNX_LOGE("Empty list of embeddings");
      return false;
    }

    for (const auto &x : embedding_list) {
      if (static_cast<int32_t>(x.size()) != dim_) {
        SHERPA_ONNX_LOGE("Given dim: %d, expected dim: %d",
                         static_cast<int32_t>(x.size()), dim_);
        return false;
      }
    }

    // compute the average
    Eigen::RowVectorXf v = Eigen::Map<Eigen::RowVectorXf>(
        const_cast<float *>(embedding_list[0].data()), dim_);
    int32_t i = -1;
    for (const auto &x : embedding_list) {
      ++i;
      if (i == 0) {
        continue;
      }
      v += Eigen::Map<Eigen::RowVectorXf>(const_cast<float *>(x.data()), dim_);
    }

    // no need to compute the mean since we are going to normalize it anyway
    // v /= embedding_list.size();

    v.normalize();

    embedding_matrix_.conservativeResize(embedding_matrix_.rows() + 1, dim_);
    embedding_matrix_.bottomRows(1) = v;

    name2row_[name] = embedding_matrix_.rows() - 1;
    row2name_[embedding_matrix_.rows() - 1] = name;

    return true;
  }

  bool Remove(const std::string &name) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    int32_t num_rows = embedding_matrix_.rows();

    if (row_idx < num_rows - 1) {
      embedding_matrix_.block(row_idx, 0, num_rows - 1 - row_idx, dim_) =
          embedding_matrix_.bottomRows(num_rows - 1 - row_idx);
    }

    embedding_matrix_.conservativeResize(num_rows - 1, dim_);
    for (auto &p : name2row_) {
      if (p.second > row_idx) {
        p.second -= 1;
        row2name_[p.second] = p.first;
      }
    }

    name2row_.erase(name);
    row2name_.erase(num_rows - 1);

    return true;
  }

  std::string Search(const float *p, float threshold) {
    if (embedding_matrix_.rows() == 0) {
      return {};
    }

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    Eigen::VectorXf scores = embedding_matrix_ * v;

    Eigen::VectorXf::Index max_index = 0;
    float max_score = scores.maxCoeff(&max_index);
    if (max_score < threshold) {
      return {};
    }

    return row2name_.at(max_index);
  }

  std::vector<SpeakerMatch> GetBestMatches(const float *p, float threshold,
                                           int32_t n) {
    std::vector<SpeakerMatch> matches;

    if (embedding_matrix_.rows() == 0) {
      return matches;
    }

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    Eigen::VectorXf scores = embedding_matrix_ * v;

    std::vector<std::pair<float, int>> score_indices;
    for (int i = 0; i < scores.size(); ++i) {
      if (scores[i] >= threshold) {
        score_indices.emplace_back(scores[i], i);
      }
    }

    std::sort(score_indices.rbegin(), score_indices.rend(),
              [](const auto &a, const auto &b) { return a.first < b.first; });

    matches.reserve(score_indices.size());
    for (int i = 0; i < std::min(n, static_cast<int32_t>(score_indices.size()));
         ++i) {
      const auto &pair = score_indices[i];
      matches.push_back({row2name_.at(pair.second), pair.first});
    }

    return matches;
  }

  bool Verify(const std::string &name, const float *p, float threshold) {
    if (!name2row_.count(name)) {
      return false;
    }

    int32_t row_idx = name2row_.at(name);

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    float score = embedding_matrix_.row(row_idx) * v;

    if (score < threshold) {
      return false;
    }

    return true;
  }

  float Score(const std::string &name, const float *p) {
    if (!name2row_.count(name)) {
      // Setting a default value if the name is not found
      return -2.0;
    }

    int32_t row_idx = name2row_.at(name);

    Eigen::VectorXf v =
        Eigen::Map<Eigen::VectorXf>(const_cast<float *>(p), dim_);
    v.normalize();

    float score = embedding_matrix_.row(row_idx) * v;

    return score;
  }

  bool Contains(const std::string &name) const {
    return name2row_.count(name) > 0;
  }

  int32_t NumSpeakers() const { return embedding_matrix_.rows(); }

  int32_t Dim() const { return dim_; }

  std::vector<std::string> GetAllSpeakers() const {
    std::vector<std::string> all_speakers;
    all_speakers.reserve(name2row_.size());
    for (const auto &p : name2row_) {
      all_speakers.push_back(p.first);
    }

    std::sort(all_speakers.begin(), all_speakers.end());
    return all_speakers;
  }

 private:
  int32_t dim_;
  FloatMatrix embedding_matrix_;
  std::unordered_map<std::string, int32_t> name2row_;
  std::unordered_map<int32_t, std::string> row2name_;
};
```

**关键发现：**
- 没有任何 `std::mutex`、`std::lock_guard` 或其他同步原语
- 没有 `#include <mutex>` 头文件
- 所有方法直接操作共享数据结构，没有任何保护

#### 1.2 方法签名分析

所有公共方法都是 `const` 方法，但这只是 C++ 的语法约定，并不表示线程安全：

```239:286:sherpa-onnx/csrc/speaker-embedding-manager.cc
bool SpeakerEmbeddingManager::Add(const std::string &name,
                                  const float *p) const {
  return impl_->Add(name, p);
}

bool SpeakerEmbeddingManager::Add(
    const std::string &name,
    const std::vector<std::vector<float>> &embedding_list) const {
  return impl_->Add(name, embedding_list);
}

bool SpeakerEmbeddingManager::Remove(const std::string &name) const {
  return impl_->Remove(name);
}

std::string SpeakerEmbeddingManager::Search(const float *p,
                                            float threshold) const {
  return impl_->Search(p, threshold);
}

std::vector<SpeakerMatch> SpeakerEmbeddingManager::GetBestMatches(
    const float *p, float threshold, int32_t n) const {
  return impl_->GetBestMatches(p, threshold, n);
}

bool SpeakerEmbeddingManager::Verify(const std::string &name, const float *p,
                                     float threshold) const {
  return impl_->Verify(name, p, threshold);
}

float SpeakerEmbeddingManager::Score(const std::string &name,
                                     const float *p) const {
  return impl_->Score(name, p);
}

int32_t SpeakerEmbeddingManager::NumSpeakers() const {
  return impl_->NumSpeakers();
}

int32_t SpeakerEmbeddingManager::Dim() const { return impl_->Dim(); }

bool SpeakerEmbeddingManager::Contains(const std::string &name) const {
  return impl_->Contains(name);
}

std::vector<std::string> SpeakerEmbeddingManager::GetAllSpeakers() const {
  return impl_->GetAllSpeakers();
}
```

### 2. 语言绑定分析

#### 2.1 Java 绑定

```1:77:sherpa-onnx/java-api/src/main/java/com/k2fsa/sherpa/onnx/SpeakerEmbeddingManager.java
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SpeakerEmbeddingManager {
    private long ptr = 0;

    public SpeakerEmbeddingManager(int dim) {
        LibraryLoader.maybeLoad();
        ptr = create(dim);
    }

    @Override
    protected void finalize() throws Throwable {
        release();
    }

    public void release() {
        if (this.ptr == 0) {
            return;
        }
        delete(this.ptr);
        this.ptr = 0;
    }

    public boolean add(String name, float[] embedding) {
        return add(ptr, name, embedding);
    }

    public boolean add(String name, float[][] embedding) {
        return addList(ptr, name, embedding);
    }

    public boolean remove(String name) {
        return remove(ptr, name);
    }

    public String search(float[] embedding, float threshold) {
        return search(ptr, embedding, threshold);
    }

    public boolean verify(String name, float[] embedding, float threshold) {
        return verify(ptr, name, embedding, threshold);
    }

    public boolean contains(String name) {
        return contains(ptr, name);
    }

    public int getNumSpeakers() {
        return numSpeakers(ptr);
    }

    public String[] getAllSpeakerNames() {
        return allSpeakerNames(ptr);
    }

    private native long create(int dim);

    private native void delete(long ptr);

    private native boolean add(long ptr, String name, float[] embedding);

    private native boolean addList(long ptr, String name, float[][] embedding);

    private native boolean remove(long ptr, String name);

    private native String search(long ptr, float[] embedding, float threshold);

    private native boolean verify(long ptr, String name, float[] embedding, float threshold);

    private native boolean contains(long ptr, String name);

    private native int numSpeakers(long ptr);

    private native String[] allSpeakerNames(long ptr);
}
```

**分析结果：**
- ❌ 没有 `synchronized` 关键字
- ❌ 没有使用 `java.util.concurrent` 包中的锁机制
- ❌ 所有方法都是非同步的

#### 2.2 Kotlin 绑定

```64:114:sherpa-onnx/kotlin-api/Speaker.kt
class SpeakerEmbeddingManager(val dim: Int) {
    private var ptr: Long

    init {
        ptr = create(dim)
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()
    fun add(name: String, embedding: FloatArray) = add(ptr, name, embedding)
    fun add(name: String, embedding: Array<FloatArray>) = addList(ptr, name, embedding)
    fun remove(name: String) = remove(ptr, name)
    fun search(embedding: FloatArray, threshold: Float) = search(ptr, embedding, threshold)
    fun verify(name: String, embedding: FloatArray, threshold: Float) =
        verify(ptr, name, embedding, threshold)

    fun contains(name: String) = contains(ptr, name)
    fun numSpeakers() = numSpeakers(ptr)

    fun allSpeakerNames() = allSpeakerNames(ptr)

    private external fun create(dim: Int): Long
    private external fun delete(ptr: Long): Unit
    private external fun add(ptr: Long, name: String, embedding: FloatArray): Boolean
    private external fun addList(ptr: Long, name: String, embedding: Array<FloatArray>): Boolean
    private external fun remove(ptr: Long, name: String): Boolean
    private external fun search(ptr: Long, embedding: FloatArray, threshold: Float): String
    private external fun verify(
        ptr: Long,
        name: String,
        embedding: FloatArray,
        threshold: Float
    ): Boolean

    private external fun contains(ptr: Long, name: String): Boolean
    private external fun numSpeakers(ptr: Long): Int

    private external fun allSpeakerNames(ptr: Long): Array<String>

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
```

**分析结果：**
- ❌ 没有使用 `@Synchronized` 注解
- ❌ 没有使用 Kotlin 的 `Mutex` 或 `ReentrantLock`
- ❌ 所有方法都是非同步的

**注意：** 在同一个文件中的 `SpeakerRecognition` 对象使用了 `synchronized(this)`，但这是用于初始化 `SpeakerEmbeddingExtractor` 和 `SpeakerEmbeddingManager` 的，而不是用于保护 `SpeakerEmbeddingManager` 的方法调用。

#### 2.3 Python 绑定

Python 绑定通过 pybind11 实现，也没有额外的同步机制。虽然使用了 `py::gil_scoped_release`，但这只是为了释放 GIL 以提高性能，并不提供线程安全保护。

#### 2.4 其他语言绑定

- **Go**: 没有使用 `sync.Mutex` 或 `sync.RWMutex`
- **C#**: 没有使用 `lock` 语句或 `Monitor`
- **JavaScript/Node.js**: 没有使用任何同步机制

## 并发问题分析

### 1. 数据竞争（Data Race）

#### 1.1 Add 方法的并发问题

**场景：** 两个线程同时调用 `Add` 方法添加不同的说话人

```cpp
// 线程 1: Add("speaker1", embedding1)
// 线程 2: Add("speaker2", embedding2)
```

**问题：**
1. 两个线程可能同时检查 `name2row_.count(name)`，都通过检查
2. 两个线程同时调用 `embedding_matrix_.conservativeResize()`，导致矩阵大小计算错误
3. 两个线程同时修改 `name2row_` 和 `row2name_`，导致映射关系不一致
4. 可能导致内存损坏或程序崩溃

#### 1.2 Remove 方法的并发问题

**场景：** 一个线程调用 `Remove`，另一个线程同时调用 `Search` 或 `Add`

```cpp
// 线程 1: Remove("speaker1")
// 线程 2: Search(embedding) 或 Add("speaker2", embedding)
```

**问题：**
1. `Remove` 方法会遍历并修改 `name2row_` map，同时 `Search` 或 `Add` 也在访问这个 map
2. `Remove` 会移动 `embedding_matrix_` 的行，而 `Search` 正在读取矩阵
3. 可能导致：
   - 读取到已删除的说话人数据
   - 读取到错误位置的嵌入向量
   - map 迭代器失效导致崩溃

#### 1.3 读写并发问题

**场景：** 多个线程同时进行读操作（Search、Verify）和写操作（Add、Remove）

**问题：**
1. **Eigen 矩阵操作不是线程安全的**：`embedding_matrix_ * v` 在矩阵被修改时可能导致未定义行为
2. **std::unordered_map 不是线程安全的**：并发读写会导致数据竞争
3. **矩阵 resize 操作**：在读取过程中 resize 矩阵会导致内存访问越界

### 2. 具体风险场景

#### 场景 1：并发 Add 操作

```python
# Python 示例
import threading
manager = SpeakerEmbeddingManager(dim=512)

def add_speaker(name, embedding):
    manager.add(name, embedding)

# 两个线程同时添加
thread1 = threading.Thread(target=add_speaker, args=("speaker1", emb1))
thread2 = threading.Thread(target=add_speaker, args=("speaker2", emb2))
thread1.start()
thread2.start()
thread1.join()
thread2.join()

# 风险：数据竞争，可能导致：
# - 矩阵大小错误
# - 映射关系错乱
# - 程序崩溃
```

#### 场景 2：并发读写

```java
// Java 示例
SpeakerEmbeddingManager manager = new SpeakerEmbeddingManager(512);

// 线程 1：添加说话人
new Thread(() -> {
    manager.add("speaker1", embedding1);
}).start();

// 线程 2：搜索说话人
new Thread(() -> {
    String result = manager.search(embedding2, 0.5f);
}).start();

// 风险：读取到不一致的数据或崩溃
```

#### 场景 3：Remove 与 Search 并发

```kotlin
// Kotlin 示例
val manager = SpeakerEmbeddingManager(dim = 512)

// 线程 1：删除说话人
thread {
    manager.remove("speaker1")
}

// 线程 2：搜索说话人
thread {
    val result = manager.search(embedding, 0.5f)
}

// 风险：
// - Search 可能访问到已删除的数据
// - 矩阵索引错误
// - 程序崩溃
```

### 3. 内存安全问题

由于没有同步机制，以下操作可能导致严重的内存安全问题：

1. **矩阵 resize 时的内存访问**：`conservativeResize` 在并发情况下可能导致内存越界访问
2. **Map 迭代器失效**：`Remove` 方法中的 map 遍历在并发修改时会导致迭代器失效
3. **指针悬空**：在 C API 中，返回的字符串指针可能在释放后被访问

## 代码库中的锁使用情况

为了对比，我们检查了代码库中其他组件的锁使用情况：

### 有锁机制的组件示例

1. **VAD 相关组件**：使用了 `std::mutex` 保护音频缓冲区
2. **TTS 播放组件**：使用了 `std::mutex` 保护音频缓冲区
3. **Swift API**：使用了 `NSLock` 保护流替换操作

**结论：** 代码库中其他组件在需要线程安全的地方都使用了锁，但 `SpeakerEmbeddingManager` 没有。

## 使用建议

### 1. 单线程使用（推荐）

**最简单安全的方案：** 确保 `SpeakerEmbeddingManager` 实例只在一个线程中使用。

```python
# Python 示例：单线程使用
manager = SpeakerEmbeddingManager(dim=512)
manager.add("speaker1", embedding1)
result = manager.search(embedding2, 0.5f)
```

### 2. 外部同步（多线程使用）

如果需要多线程访问，必须在外部添加同步机制：

#### Python 示例

```python
import threading

class ThreadSafeSpeakerEmbeddingManager:
    def __init__(self, dim):
        self._manager = SpeakerEmbeddingManager(dim)
        self._lock = threading.RLock()  # 使用可重入锁
    
    def add(self, name, embedding):
        with self._lock:
            return self._manager.add(name, embedding)
    
    def remove(self, name):
        with self._lock:
            return self._manager.remove(name)
    
    def search(self, embedding, threshold):
        with self._lock:
            return self._manager.search(embedding, threshold)
    
    def verify(self, name, embedding, threshold):
        with self._lock:
            return self._manager.verify(name, embedding, threshold)
    
    # ... 其他方法类似
```

#### Java 示例

```java
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class ThreadSafeSpeakerEmbeddingManager {
    private final SpeakerEmbeddingManager manager;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    
    public ThreadSafeSpeakerEmbeddingManager(int dim) {
        this.manager = new SpeakerEmbeddingManager(dim);
    }
    
    public boolean add(String name, float[] embedding) {
        lock.writeLock().lock();
        try {
            return manager.add(name, embedding);
        } finally {
            lock.writeLock().unlock();
        }
    }
    
    public String search(float[] embedding, float threshold) {
        lock.readLock().lock();
        try {
            return manager.search(embedding, threshold);
        } finally {
            lock.readLock().unlock();
        }
    }
    
    // ... 其他方法类似
    // 注意：Add/Remove 使用 writeLock，Search/Verify 使用 readLock
}
```

#### Kotlin 示例

```kotlin
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

class ThreadSafeSpeakerEmbeddingManager(dim: Int) {
    private val manager = SpeakerEmbeddingManager(dim)
    private val mutex = Mutex()
    
    suspend fun add(name: String, embedding: FloatArray): Boolean {
        return mutex.withLock {
            manager.add(name, embedding)
        }
    }
    
    suspend fun search(embedding: FloatArray, threshold: Float): String {
        return mutex.withLock {
            manager.search(embedding, threshold)
        }
    }
    
    // ... 其他方法类似
}
```

### 3. 读写锁优化

如果读操作（Search、Verify）远多于写操作（Add、Remove），可以使用读写锁提高并发性能：

```cpp
// C++ 示例（需要修改源码）
#include <shared_mutex>

class ThreadSafeSpeakerEmbeddingManager {
private:
    SpeakerEmbeddingManager manager_;
    mutable std::shared_mutex mutex_;
    
public:
    bool Add(const std::string &name, const float *p) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return manager_.Add(name, p);
    }
    
    std::string Search(const float *p, float threshold) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return manager_.Search(p, threshold);
    }
    
    // ... 其他方法类似
};
```

### 4. 避免的操作

**绝对不要：**
- ❌ 在多线程中直接使用 `SpeakerEmbeddingManager` 而不加锁
- ❌ 在回调函数中不加锁地访问 `SpeakerEmbeddingManager`
- ❌ 在异步操作中不加锁地访问 `SpeakerEmbeddingManager`

## 性能影响

### 1. 加锁的性能开销

- **互斥锁（Mutex）**：每次操作都有锁获取/释放的开销，约 10-100 纳秒
- **读写锁（ReadWriteLock）**：读操作可以并发，写操作需要独占，适合读多写少的场景
- **无锁方案**：需要修改核心实现，使用原子操作或无锁数据结构

### 2. 性能优化建议

1. **批量操作**：尽量批量添加说话人，减少锁的获取次数
2. **读写分离**：使用读写锁，允许多个读操作并发
3. **本地缓存**：对于频繁查询的说话人，可以在应用层缓存结果

## 改进建议

### 1. 短期方案（应用层）

在应用层添加同步机制，如上面的示例代码。

### 2. 长期方案（修改源码）

如果需要原生支持线程安全，建议修改核心实现：

1. **在 `Impl` 类中添加互斥锁**：
```cpp
class SpeakerEmbeddingManager::Impl {
private:
    mutable std::shared_mutex mutex_;  // 读写锁
    // ... 其他成员
};
```

2. **在所有方法中添加锁保护**：
```cpp
bool Add(const std::string &name, const float *p) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    // ... 原有逻辑
}
```

3. **考虑使用无锁数据结构**（更复杂但性能更好）

## 总结

| 项目 | 状态 |
|------|------|
| 核心 C++ 实现是否有锁 | ❌ 否 |
| Java 绑定是否有同步 | ❌ 否 |
| Kotlin 绑定是否有同步 | ❌ 否 |
| Python 绑定是否有同步 | ❌ 否 |
| 其他语言绑定是否有同步 | ❌ 否 |
| 是否线程安全 | ❌ 否 |
| 并发使用是否安全 | ❌ 否 |

**关键结论：**
- `SpeakerEmbeddingManager` **不是线程安全的**
- **不应该在多线程环境中直接使用**，除非在外部添加同步机制
- 建议在应用层实现线程安全的包装类，或修改源码添加锁机制

## 参考资料

- 核心实现：`sherpa-onnx/csrc/speaker-embedding-manager.cc`
- 头文件：`sherpa-onnx/csrc/speaker-embedding-manager.h`
- Java 绑定：`sherpa-onnx/java-api/src/main/java/com/k2fsa/sherpa/onnx/SpeakerEmbeddingManager.java`
- Kotlin 绑定：`sherpa-onnx/kotlin-api/Speaker.kt`
- C API：`sherpa-onnx/c-api/c-api.cc`
