#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

/****************************************************************
* Class Name: ThreadPool
* Description: 고정된 스레드 수로 작업 큐를 처리하는 스레드 풀 구현
****************************************************************/
class ThreadPool {
private:
    std::vector<std::thread> workers;       // 작업을 처리하는 스레드 배열
    std::queue<std::function<void()>> tasks;// 작업 큐
    mutable std::mutex queueMutex;          // 큐 접근을 동기화하는 뮤텍스 (const 메서드에서도 사용 가능)
    std::condition_variable condition;      // 스레드 대기를 위한 조건 변수
    bool stop;                              // 스레드 풀 종료 플래그

public:
    // 지정된 스레드 수로 스레드 풀을 초기화하고 작업 대기 시작
    explicit ThreadPool(size_t numThreads);


    // 스레드 풀을 종료하고 모든 스레드가 완료될 때까지 대기
    ~ThreadPool();

    // 작업 큐에 새 작업을 추가하고 스레드를 깨움
    void Enqueue(std::function<void()> task);

    // 현재 작업 큐에 남아 있는 작업 수 반환
    [[nodiscard]] size_t GetTaskCount() const;

    // 작업 큐가 비어 있는지 확인
    [[nodiscard]] bool IsEmpty() const;
};
