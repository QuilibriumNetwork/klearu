//! Transport implementation for WASM Web Workers.
//!
//! Uses SharedArrayBuffer + Atomics for recv (main thread writes data,
//! worker blocks on Atomics.wait) and JS callbacks (postMessage) for send.
//!
//! SAB layout:
//!   [0..4]   : i32 control — 0=empty, 1=data_ready
//!   [4..8]   : u32 data length (LE)
//!   [8..N]   : data payload bytes
//!
//! Main thread → worker (recv): write data at offset 8, length at 4,
//!   store 1 at offset 0, Atomics.notify.
//! Worker → main thread (send): postMessage({type: "mpc_send", data}).
//! Worker after consuming: store 0 at offset 0 (so main thread knows
//!   buffer is free before writing more).

use js_sys::{Atomics, Int32Array, Object, Reflect, SharedArrayBuffer, Uint8Array};
use klearu_mpc::transport::Transport;
use std::io;
use wasm_bindgen::prelude::*;

/// Transport that bridges WASM worker ↔ main thread via SharedArrayBuffer.
pub struct WasmTransport {
    /// SharedArrayBuffer for recv (main thread writes, worker reads).
    sab: SharedArrayBuffer,
    /// i32 view over the control word at offset 0.
    control: Int32Array,
    /// JS callback for send — calls postMessage to main thread.
    send_callback: js_sys::Function,
    /// Internal recv buffer for handling partial reads.
    recv_buf: Vec<u8>,
}

impl WasmTransport {
    /// Create a new WasmTransport.
    ///
    /// `sab`: SharedArrayBuffer (must be at least 1MB for typical MPC messages).
    /// `send_callback`: JS function that accepts a JsValue message object
    ///   and posts it to the main thread.
    pub fn new(sab: SharedArrayBuffer, send_callback: js_sys::Function) -> Self {
        // Create an Int32Array view over just the first 4 bytes (control word)
        let control = Int32Array::new_with_byte_offset_and_length(&sab, 0, 1);
        Self {
            sab,
            control,
            send_callback,
            recv_buf: Vec::new(),
        }
    }

    /// Block until the main thread signals data is ready, then read it.
    fn recv_from_sab(&mut self) -> io::Result<Vec<u8>> {
        // Wait for control word to become 1 (data_ready)
        // Atomics.wait blocks the thread until notified or value changes
        loop {
            let val = Atomics::load(&self.control, 0)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

            if val == 1 {
                break;
            }

            // Block until notified — wait expects value 0, wakes when it changes
            let _ = Atomics::wait(&self.control, 0, 0);
        }

        // Read length from bytes 4..8
        let len_view = Uint8Array::new_with_byte_offset_and_length(&self.sab, 4, 4);
        let mut len_bytes = [0u8; 4];
        len_view.copy_to(&mut len_bytes);
        let data_len = u32::from_le_bytes(len_bytes) as usize;

        // Read data from bytes 8..8+data_len
        let data_view = Uint8Array::new_with_byte_offset_and_length(&self.sab, 8, data_len as u32);
        let mut data = vec![0u8; data_len];
        data_view.copy_to(&mut data);

        // Signal that we've consumed the data (set control to 0)
        Atomics::store(&self.control, 0, 0)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        // Notify main thread that buffer is free
        let _ = Atomics::notify(&self.control, 0);

        Ok(data)
    }
}

impl Transport for WasmTransport {
    fn send(&mut self, data: &[u8]) -> io::Result<()> {
        let js_data = Uint8Array::from(data);

        // Build message: {type: "mpc_send", data: Uint8Array}
        let msg = Object::new();
        Reflect::set(&msg, &JsValue::from_str("type"), &JsValue::from_str("mpc_send"))
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;
        Reflect::set(&msg, &JsValue::from_str("data"), &js_data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        self.send_callback
            .call1(&JsValue::NULL, &msg)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))?;

        Ok(())
    }

    fn recv(&mut self, len: usize) -> io::Result<Vec<u8>> {
        // Drain from internal buffer first
        while self.recv_buf.len() < len {
            let chunk = self.recv_from_sab()?;
            self.recv_buf.extend_from_slice(&chunk);
        }
        Ok(self.recv_buf.drain(..len).collect())
    }
}
