#ifndef MANILLE_PARALLEL_H
#define MANILLE_PARALLEL_H

// Lightweight persistent worker pool for data-parallel play-out loops
// (cpp/fast_policy.cpp, cpp/rule_playout.cpp).
//
// Properties:
//  - Uses up to 4 threads (3 workers + caller) by default, matching the big
//    cores on modern phones and desktops without straggler stalls.
//  - Non-blocking re-entrancy: if another outer thread already holds the pool
//    (e.g. a multi-threaded benchmark harness), falls back inline with zero
//    contention or deadlock.
//  - Pure C++17 (<thread>, <mutex>, <condition_variable>, <atomic>): portable
//    across Android ARM64, iOS, Windows MSVC, macOS Apple Silicon, and Linux.

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace manille_parallel {

class Pool {
public:
    static Pool& instance() {
        static Pool p;
        return p;
    }

    void set_max_threads(int n) {
        m_max_threads.store(std::max(1, n), std::memory_order_relaxed);
    }

    int max_threads() const {
        return m_max_threads.load(std::memory_order_relaxed);
    }

    // Runs fn(begin, end) over [0, count) split into contiguous chunks.
    // Falls back to single-threaded execution when count < min_items,
    // max_threads <= 1, or the pool is already busy.
    template <typename Fn>
    void parallel_for(size_t count, size_t min_items, const Fn& fn) {
        if (count == 0)
            return;
        const int cap = m_max_threads.load(std::memory_order_relaxed);
        if (cap <= 1 || count < min_items || m_workers.empty()) {
            fn(0, count);
            return;
        }
        std::unique_lock<std::mutex> run_lk(m_run_mu, std::try_to_lock);
        if (!run_lk.owns_lock()) {
            fn(0, count);
            return;
        }
        const size_t n_threads = std::min<size_t>(
            static_cast<size_t>(cap),
            std::min<size_t>(m_workers.size() + 1, (count + 3) / 4));
        if (n_threads <= 1) {
            fn(0, count);
            return;
        }

        std::function<void(size_t, size_t)> task = [&](size_t b, size_t e) { fn(b, e); };
        const size_t n_helpers = n_threads - 1;
        {
            std::lock_guard<std::mutex> lk(m_mu);
            m_task = &task;
            m_count = count;
            m_n_threads = n_threads;
            m_active_helpers = n_helpers;
            m_done_helpers = 0;
            ++m_gen;
        }
        m_cv.notify_all();

        // Calling thread executes chunk 0.
        const size_t b0 = 0;
        const size_t e0 = count / n_threads;
        if (b0 < e0)
            fn(b0, e0);

        std::unique_lock<std::mutex> lk(m_mu);
        m_done_cv.wait(lk, [&] { return m_done_helpers == n_helpers; });
        m_task = nullptr;
    }

private:
    Pool() {
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned total = std::min(4u, std::max(1u, hw));
        m_max_threads.store(static_cast<int>(total), std::memory_order_relaxed);
        const unsigned n_workers = total > 1 ? total - 1 : 0;
        m_workers.reserve(n_workers);
        for (unsigned id = 1; id <= n_workers; ++id) {
            m_workers.emplace_back([this, id] { workerLoop(id); });
        }
    }

    ~Pool() {
        {
            std::lock_guard<std::mutex> lk(m_mu);
            m_stop = true;
            ++m_gen;
        }
        m_cv.notify_all();
        for (auto& t : m_workers) {
            if (t.joinable())
                t.join();
        }
    }

    void workerLoop(unsigned id) {
        uint64_t seen_gen = 0;
        for (;;) {
            const std::function<void(size_t, size_t)>* task = nullptr;
            size_t count = 0;
            size_t n_threads = 1;
            bool participate = false;
            {
                std::unique_lock<std::mutex> lk(m_mu);
                m_cv.wait(lk, [&] { return m_gen != seen_gen; });
                seen_gen = m_gen;
                if (m_stop)
                    return;
                if (id <= m_active_helpers && m_task != nullptr) {
                    participate = true;
                    task = m_task;
                    count = m_count;
                    n_threads = m_n_threads;
                }
            }
            if (participate) {
                const size_t b = (count * id) / n_threads;
                const size_t e = (count * (id + 1)) / n_threads;
                if (b < e)
                    (*task)(b, e);
                std::lock_guard<std::mutex> lk(m_mu);
                ++m_done_helpers;
                if (m_done_helpers == m_active_helpers)
                    m_done_cv.notify_one();
            }
        }
    }

    std::mutex m_run_mu;
    std::mutex m_mu;
    std::condition_variable m_cv;
    std::condition_variable m_done_cv;
    std::vector<std::thread> m_workers;
    std::atomic<int> m_max_threads{1};
    const std::function<void(size_t, size_t)>* m_task = nullptr;
    size_t m_count = 0;
    size_t m_n_threads = 1;
    size_t m_active_helpers = 0;
    size_t m_done_helpers = 0;
    uint64_t m_gen = 0;
    bool m_stop = false;
};

template <typename Fn>
inline void parallel_for(size_t count, size_t min_items, const Fn& fn) {
    Pool::instance().parallel_for(count, min_items, fn);
}

inline void set_max_threads(int n) {
    Pool::instance().set_max_threads(n);
}

} // namespace manille_parallel

#endif // MANILLE_PARALLEL_H
