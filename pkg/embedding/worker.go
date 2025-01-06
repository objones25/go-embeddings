package embedding

import (
	"context"
	"fmt"
	"sync"

	"go.uber.org/zap"
)

// WorkerPool manages a pool of workers for parallel embedding processing
type WorkerPool struct {
	workers    int
	queue      chan Task
	logger     *zap.Logger
	wg         sync.WaitGroup
	cancelFunc context.CancelFunc
}

// Task represents a single embedding task
type Task struct {
	Text   string
	Result chan<- Result
	Error  chan<- error
}

// NewWorkerPool creates a new worker pool with the specified number of workers
func NewWorkerPool(workers, queueSize int) *WorkerPool {
	ctx, cancel := context.WithCancel(context.Background())
	logger, _ := zap.NewProduction()

	pool := &WorkerPool{
		workers:    workers,
		queue:      make(chan Task, queueSize),
		logger:     logger,
		cancelFunc: cancel,
	}

	pool.start(ctx)
	return pool
}

// start initializes the worker pool
func (p *WorkerPool) start(ctx context.Context) {
	for i := 0; i < p.workers; i++ {
		p.wg.Add(1)
		go p.worker(ctx, i)
	}
}

// worker processes tasks from the queue
func (p *WorkerPool) worker(ctx context.Context, id int) {
	defer p.wg.Done()

	p.logger.Info("Starting worker", zap.Int("worker_id", id))

	for {
		select {
		case <-ctx.Done():
			p.logger.Info("Worker shutting down", zap.Int("worker_id", id))
			return
		case task, ok := <-p.queue:
			if !ok {
				return
			}
			// Process the task
			p.processTask(ctx, task, id)
		}
	}
}

// processTask handles the embedding generation for a single task
func (p *WorkerPool) processTask(ctx context.Context, task Task, workerID int) {
	select {
	case <-ctx.Done():
		select {
		case task.Error <- ctx.Err():
		default:
		}
		return
	default:
	}

	p.logger.Debug("Processing task",
		zap.Int("worker_id", workerID),
		zap.String("text", task.Text))

	// Task processing will be implemented by the embedding service
}

// Submit adds a task to the worker pool
func (p *WorkerPool) Submit(task Task) error {
	select {
	case p.queue <- task:
		return nil
	default:
		return fmt.Errorf("worker queue is full")
	}
}

// Shutdown gracefully shuts down the worker pool
func (p *WorkerPool) Shutdown() {
	p.cancelFunc()
	close(p.queue)
	p.wg.Wait()
}
