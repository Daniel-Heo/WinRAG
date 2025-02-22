/*******************************************************************************
    파   일   명 : thread_pool.cpp
	프로그램명칭 :  쓰레드 풀 클래스
    작   성   일 : 2025.2.22
    작   성   자 : Daniel Heo ( https://github.com/Daniel-Heo/WinRAG )
	프로그램용도 : 쓰레드 풀을 구현하여 여러 스레드에서 작업을 처리할 수 있도록 함
    참 고 사 항  :
    라 이 센 스  : MIT License

    ----------------------------------------------------------------------------
    수정일자    수정자      수정내용
    =========== =========== ====================================================
    2025.2.22   Daniel Heo  최초 생성
    ----------------------------------------------------------------------------

*******************************************************************************/
#include "thread_pool.h"

/****************************************************************
* Function Name: ThreadPool (생성자)
* Description: 지정된 스레드 수로 스레드 풀을 초기화하고 작업 대기 시작
* Parameters:
*   - numThreads: 생성할 스레드 수
* Return: 없음
* Date: 2025-02-21
****************************************************************/
ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; i++) {
        workers.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock, [this] { return stop || !tasks.empty(); });
                    if (stop && tasks.empty()) return; // 종료 조건
                    task = std::move(tasks.front());   // 작업 가져오기
                    tasks.pop();
                }
                task(); // 작업 실행
            }
            });
    }
}

/****************************************************************
* Function Name: ~ThreadPool (소멸자)
* Description: 스레드 풀을 종료하고 모든 스레드가 완료될 때까지 대기
* Parameters: 없음
* Return: 없음
* Date: 2025-02-21
****************************************************************/
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true; // 종료 신호 설정
    }
    condition.notify_all(); // 모든 스레드 깨우기
    for (auto& worker : workers) worker.join(); // 스레드 종료 대기
}

/****************************************************************
* Function Name: Enqueue
* Description: 작업 큐에 새 작업을 추가하고 스레드를 깨움
* Parameters:
*   - task: 실행할 함수 객체
* Return: 없음
* Date: 2025-02-21
****************************************************************/
void ThreadPool::Enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (stop) return; // 종료 상태면 추가하지 않음
        tasks.emplace(std::move(task));
    }
    condition.notify_one(); // 대기 중인 스레드 하나 깨우기
}

/****************************************************************
* Function Name: GetTaskCount
* Description: 현재 작업 큐에 남아 있는 작업 수 반환
* Parameters: 없음
* Return: 작업 수 (size_t)
* Date: 2025-02-21
****************************************************************/
size_t ThreadPool::GetTaskCount() const {
    std::unique_lock<std::mutex> lock(queueMutex);
    return tasks.size();
}

/****************************************************************
* Function Name: IsEmpty
* Description: 작업 큐가 비어 있는지 확인
* Parameters: 없음
* Return: 비어 있으면 true, 아니면 false
* Date: 2025-02-21
****************************************************************/
bool ThreadPool::IsEmpty() const {
    std::unique_lock<std::mutex> lock(queueMutex);
    return tasks.empty();
}
