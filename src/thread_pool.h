#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <functional>
#include <condition_variable>

/****************************************************************
* Class Name: ThreadPool
* Description: ������ ������ ���� �۾� ť�� ó���ϴ� ������ Ǯ ����
****************************************************************/
class ThreadPool {
private:
    std::vector<std::thread> workers;       // �۾��� ó���ϴ� ������ �迭
    std::queue<std::function<void()>> tasks;// �۾� ť
    mutable std::mutex queueMutex;          // ť ������ ����ȭ�ϴ� ���ؽ� (const �޼��忡���� ��� ����)
    std::condition_variable condition;      // ������ ��⸦ ���� ���� ����
    bool stop;                              // ������ Ǯ ���� �÷���

public:
    // ������ ������ ���� ������ Ǯ�� �ʱ�ȭ�ϰ� �۾� ��� ����
    explicit ThreadPool(size_t numThreads);


    // ������ Ǯ�� �����ϰ� ��� �����尡 �Ϸ�� ������ ���
    ~ThreadPool();

    // �۾� ť�� �� �۾��� �߰��ϰ� �����带 ����
    void Enqueue(std::function<void()> task);

    // ���� �۾� ť�� ���� �ִ� �۾� �� ��ȯ
    [[nodiscard]] size_t GetTaskCount() const;

    // �۾� ť�� ��� �ִ��� Ȯ��
    [[nodiscard]] bool IsEmpty() const;
};
