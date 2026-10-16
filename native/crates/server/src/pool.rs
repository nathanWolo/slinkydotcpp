//! The search threads: a fixed pool (one thread per core by default), each with its own search
//! workspace, taking `/move` jobs in arrival order. A request's search runs on one thread, so
//! as many games as there are threads are searched in parallel.

use crate::engine::{Choice, Snake};
use crate::info;
use slinky_api::Request;
use slinky_search::Search;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;
use tokio::sync::oneshot;

/// One `/move` to answer.
pub struct Job {
    /// The snake asked.
    pub snake: Arc<Snake>,
    /// The parsed request.
    pub req: Arc<Request>,
    /// When the request reached the handler (a time budget counts from here).
    pub t_in: Instant,
    /// When the search must stop (the request's deadline, see `http`).
    pub deadline: Option<Instant>,
    /// Where the answer goes. A job whose receiver is gone (the handler already answered with
    /// the safe move) is skipped.
    pub reply: oneshot::Sender<Choice>,
}

/// The search thread pool.
pub struct Pool {
    tx: Mutex<mpsc::Sender<Job>>,
    threads: usize,
}

impl Pool {
    /// Starts `threads` search threads (at least one).
    pub fn new(threads: usize) -> std::io::Result<Pool> {
        let threads = threads.max(1);
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        for i in 0..threads {
            let rx = Arc::clone(&rx);
            std::thread::Builder::new()
                .name(format!("search-{i}"))
                .stack_size(8 << 20)
                .spawn(move || work(&rx))?;
        }
        Ok(Pool {
            tx: Mutex::new(tx),
            threads,
        })
    }

    /// The number of search threads.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Queues a job; gives it back if the pool has stopped.
    pub fn submit(&self, job: Job) -> Result<(), Box<Job>> {
        let tx = self.tx.lock().unwrap_or_else(|e| e.into_inner());
        tx.send(job).map_err(|e| Box::new(e.0))
    }
}

fn work(rx: &Mutex<mpsc::Receiver<Job>>) {
    let mut workspace = Search::new();
    loop {
        let job = {
            let rx = rx.lock().unwrap_or_else(|e| e.into_inner());
            match rx.recv() {
                Ok(job) => job,
                Err(_) => return, // the server is gone
            }
        };
        if job.reply.is_closed() {
            continue;
        }
        let res = catch_unwind(AssertUnwindSafe(|| {
            job.snake
                .choose_until(&mut workspace, &job.req, job.t_in, job.deadline)
        }));
        match res {
            Ok(choice) => {
                let _ = job.reply.send(choice);
            }
            Err(_) => {
                // Only reachable when built with panic = "unwind": the handler answers with
                // the safe move when the reply is dropped.
                info!(
                    "{}: the search panicked; answering safe_move",
                    job.snake.name
                );
                workspace = Search::new();
            }
        }
    }
}
