//! Runtime-agnostic bridge from Apple completion handlers to Rust futures.
//!
//! Provides [`CompletionFuture<T>`] -- a [`Future`] that resolves when an
//! Objective-C completion handler fires. Works with any async runtime
//! (tokio, async-std, smol) or can be blocked on synchronously.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use crate::error::Result;

/// Shared state between the completion handler and the future.
struct Shared<T> {
    value: Option<Result<T>>,
    waker: Option<Waker>,
}

/// A future that resolves when an Apple completion handler fires.
///
/// Created by [`completion_channel`]. The sender half is passed into
/// the Objective-C block; the future half is returned to the caller.
pub struct CompletionFuture<T> {
    shared: Arc<Mutex<Shared<T>>>,
}

// Safety: The Arc<Mutex<>> provides thread-safe interior mutability.
// T must be Send because the completion handler fires on a GCD queue
// (different thread) and sends T to the future's polling thread.
unsafe impl<T: Send> Send for CompletionFuture<T> {}

impl<T: Send> Future for CompletionFuture<T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut shared = self.shared.lock().unwrap();
        if let Some(value) = shared.value.take() {
            Poll::Ready(value)
        } else {
            shared.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

impl<T: Send> CompletionFuture<T> {
    /// Block the current thread until the completion handler fires.
    ///
    /// This is a convenience for callers who don't have an async runtime.
    /// Uses a condvar internally -- no external dependencies required.
    pub fn block_on(self) -> Result<T> {
        // Fast path: value is already available.
        {
            let mut shared = self.shared.lock().unwrap();
            if let Some(value) = shared.value.take() {
                return value;
            }
        }

        // Slow path: wait on a condvar.
        let pair = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
        let pair_for_waker = pair.clone();

        {
            let mut shared = self.shared.lock().unwrap();
            // Re-check after acquiring lock.
            if let Some(value) = shared.value.take() {
                return value;
            }
            let waker = condvar_waker(pair_for_waker);
            shared.waker = Some(waker);
        }

        // Wait for the waker to fire.
        let (lock, cvar) = &*pair;
        let mut ready = lock.lock().unwrap();
        while !*ready {
            ready = cvar.wait(ready).unwrap();
        }

        let mut shared = self.shared.lock().unwrap();
        shared.value.take().expect("waker fired but no value was set")
    }
}

/// Creates a channel pair: a [`CompletionSender`] and a [`CompletionFuture`].
///
/// The sender is designed to be called exactly once from inside an
/// Objective-C completion handler block.
pub(crate) fn completion_channel<T: Send>() -> (CompletionSender<T>, CompletionFuture<T>) {
    let shared = Arc::new(Mutex::new(Shared {
        value: None,
        waker: None,
    }));

    let sender = CompletionSender {
        shared: shared.clone(),
    };

    let future = CompletionFuture { shared };

    (sender, future)
}

/// Sender half of the completion channel.
///
/// Call [`send()`](CompletionSender::send) from within the ObjC completion
/// handler block. Consumes self to enforce exactly-once semantics.
pub(crate) struct CompletionSender<T> {
    shared: Arc<Mutex<Shared<T>>>,
}

// Safety: CompletionSender is designed to be moved into a block2 closure
// and called on a GCD dispatch queue (different thread). The Arc<Mutex<>>
// provides thread safety.
unsafe impl<T: Send> Send for CompletionSender<T> {}
unsafe impl<T: Send> Sync for CompletionSender<T> {}

impl<T: Send> CompletionSender<T> {
    /// Send the completion result, waking the future if it's being polled.
    pub fn send(self, value: Result<T>) {
        let mut shared = self.shared.lock().unwrap();
        shared.value = Some(value);
        if let Some(waker) = shared.waker.take() {
            waker.wake();
        }
    }
}

/// Create a [`Waker`] that signals a condvar when woken.
///
/// The waker holds an `Arc` reference to the (Mutex<bool>, Condvar) pair.
/// When woken, it sets the bool to `true` and calls `notify_one()`.
fn condvar_waker(
    pair: Arc<(std::sync::Mutex<bool>, std::sync::Condvar)>,
) -> Waker {
    use std::task::{RawWaker, RawWakerVTable};

    type CondvarPair = (std::sync::Mutex<bool>, std::sync::Condvar);

    unsafe fn clone_fn(data: *const ()) -> RawWaker {
        let arc = Arc::from_raw(data as *const CondvarPair);
        let cloned = arc.clone();
        // Don't drop the original -- we borrowed it via from_raw.
        std::mem::forget(arc);
        RawWaker::new(Arc::into_raw(cloned) as *const (), &VTABLE)
    }

    unsafe fn wake_fn(data: *const ()) {
        // Takes ownership (consumes the Arc).
        let arc = Arc::from_raw(data as *const CondvarPair);
        let (lock, cvar) = &*arc;
        let mut ready = lock.lock().unwrap();
        *ready = true;
        cvar.notify_one();
        // arc drops here, decrementing refcount.
    }

    unsafe fn wake_by_ref_fn(data: *const ()) {
        // Borrows -- must not drop the Arc.
        let arc = Arc::from_raw(data as *const CondvarPair);
        {
            let (lock, cvar) = &*arc;
            let mut ready = lock.lock().unwrap();
            *ready = true;
            cvar.notify_one();
            drop(ready);
        }
        std::mem::forget(arc);
    }

    unsafe fn drop_fn(data: *const ()) {
        // Drop the Arc, decrementing refcount.
        drop(Arc::from_raw(data as *const CondvarPair));
    }

    static VTABLE: RawWakerVTable =
        RawWakerVTable::new(clone_fn, wake_fn, wake_by_ref_fn, drop_fn);

    let data = Arc::into_raw(pair) as *const ();
    // Safety: The RawWaker vtable correctly manages the Arc refcount.
    // clone increments, wake/drop decrement, wake_by_ref is neutral.
    unsafe { Waker::from_raw(RawWaker::new(data, &VTABLE)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{Error, ErrorKind};

    #[test]
    fn send_then_block_on() {
        let (sender, future) = completion_channel::<String>();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            sender.send(Ok("hello".to_string()));
        });

        let result = future.block_on().unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn error_propagation() {
        let (sender, future) = completion_channel::<String>();

        std::thread::spawn(move || {
            sender.send(Err(Error::new(ErrorKind::ModelLoad, "test error")));
        });

        let err = future.block_on().unwrap_err();
        assert_eq!(err.kind(), &ErrorKind::ModelLoad);
    }

    #[test]
    fn immediate_value() {
        let (sender, future) = completion_channel::<i32>();
        // Value is set before block_on -- exercises the fast path.
        sender.send(Ok(42));
        assert_eq!(future.block_on().unwrap(), 42);
    }

    #[test]
    fn poll_via_future_trait() {
        use std::task::{RawWaker, RawWakerVTable};

        // Minimal noop waker for manual polling.
        fn noop_waker() -> Waker {
            unsafe fn clone(_: *const ()) -> RawWaker {
                RawWaker::new(std::ptr::null(), &NOOP_VTABLE)
            }
            unsafe fn noop(_: *const ()) {}
            static NOOP_VTABLE: RawWakerVTable =
                RawWakerVTable::new(clone, noop, noop, noop);
            unsafe {
                Waker::from_raw(RawWaker::new(std::ptr::null(), &NOOP_VTABLE))
            }
        }

        let (sender, mut future) = completion_channel::<u64>();
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);

        // First poll: pending.
        let pinned = Pin::new(&mut future);
        assert!(pinned.poll(&mut cx).is_pending());

        // Send value.
        sender.send(Ok(99));

        // Second poll: ready.
        let pinned = Pin::new(&mut future);
        match pinned.poll(&mut cx) {
            Poll::Ready(Ok(v)) => assert_eq!(v, 99),
            other => panic!("expected Ready(Ok(99)), got {other:?}"),
        }
    }

    #[test]
    fn concurrent_stress() {
        // Spawn many channels concurrently to test for races.
        let handles: Vec<_> = (0..50)
            .map(|i| {
                let (sender, future) = completion_channel::<i32>();
                let h = std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_micros(i * 10));
                    sender.send(Ok(i as i32));
                });
                (h, future)
            })
            .collect();

        for (i, (handle, future)) in handles.into_iter().enumerate() {
            let val = future.block_on().unwrap();
            assert_eq!(val, i as i32);
            handle.join().unwrap();
        }
    }
}
