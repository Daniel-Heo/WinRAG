#include "thread_pool.h"

/****************************************************************
* Function Name: ThreadPool (������)
* Description: ������ ������ ���� ������ Ǯ�� �ʱ�ȭ�ϰ� �۾� ��� ����
* Parameters:
*   - numThreads: ������ ������ ��
* Return: ����
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
                    if (stop && tasks.empty()) return; // ���� ����
                    task = std::move(tasks.front());   // �۾� ��������
                    tasks.pop();
                }
                task(); // �۾� ����
            }
            });
    }
}

/****************************************************************
* Function Name: ~ThreadPool (�Ҹ���)
* Description: ������ Ǯ�� �����ϰ� ��� �����尡 �Ϸ�� ������ ���
* Parameters: ����
* Return: ����
* Date: 2025-02-21
****************************************************************/
ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true; // ���� ��ȣ ����
    }
    condition.notify_all(); // ��� ������ �����
    for (auto& worker : workers) worker.join(); // ������ ���� ���
}

/****************************************************************
* Function Name: Enqueue
* Description: �۾� ť�� �� �۾��� �߰��ϰ� �����带 ����
* Parameters:
*   - task: ������ �Լ� ��ü
* Return: ����
* Date: 2025-02-21
****************************************************************/
void ThreadPool::Enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        if (stop) return; // ���� ���¸� �߰����� ����
        tasks.emplace(std::move(task));
    }
    condition.notify_one(); // ��� ���� ������ �ϳ� �����
}

/****************************************************************
* Function Name: GetTaskCount
* Description: ���� �۾� ť�� ���� �ִ� �۾� �� ��ȯ
* Parameters: ����
* Return: �۾� �� (size_t)
* Date: 2025-02-21
****************************************************************/
size_t ThreadPool::GetTaskCount() const {
    std::unique_lock<std::mutex> lock(queueMutex);
    return tasks.size();
}

/****************************************************************
* Function Name: IsEmpty
* Description: �۾� ť�� ��� �ִ��� Ȯ��
* Parameters: ����
* Return: ��� ������ true, �ƴϸ� false
* Date: 2025-02-21
****************************************************************/
bool ThreadPool::IsEmpty() const {
    std::unique_lock<std::mutex> lock(queueMutex);
    return tasks.empty();
}
