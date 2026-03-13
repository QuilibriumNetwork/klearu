use std::io;

/// Trait for sending/receiving bytes between two parties in a 2PC protocol.
pub trait Transport {
    fn send(&mut self, data: &[u8]) -> io::Result<()>;
    fn recv(&mut self, len: usize) -> io::Result<Vec<u8>>;

    /// Receive exactly `buf.len()` bytes into a pre-allocated buffer.
    /// Default implementation calls `recv` and copies; override for zero-alloc receives.
    fn recv_into(&mut self, buf: &mut [u8]) -> io::Result<()> {
        let data = self.recv(buf.len())?;
        buf.copy_from_slice(&data);
        Ok(())
    }

    /// Receive n u32 values into a pre-allocated buffer (zero-copy on LE platforms).
    #[cfg(target_endian = "little")]
    fn recv_u32_slice_into(&mut self, buf: &mut [u32]) -> io::Result<()> {
        self.recv_into(bytemuck::cast_slice_mut(buf))
    }

    #[cfg(not(target_endian = "little"))]
    fn recv_u32_slice_into(&mut self, buf: &mut [u32]) -> io::Result<()> {
        let data = self.recv(buf.len() * 4)?;
        for (i, out) in buf.iter_mut().enumerate() {
            let offset = i * 4;
            *out = u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]);
        }
        Ok(())
    }

    /// Send a u32 value.
    fn send_u32(&mut self, v: u32) -> io::Result<()> {
        self.send(&v.to_le_bytes())
    }

    /// Receive a u32 value.
    fn recv_u32(&mut self) -> io::Result<u32> {
        let data = self.recv(4)?;
        Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]))
    }

    /// Send a slice of u32 values (zero-copy on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn send_u32_slice(&mut self, values: &[u32]) -> io::Result<()> {
        self.send(bytemuck::cast_slice(values))
    }

    #[cfg(not(target_endian = "little"))]
    fn send_u32_slice(&mut self, values: &[u32]) -> io::Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.send(&bytes)
    }

    /// Receive n u32 values (bulk cast on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn recv_u32_slice(&mut self, n: usize) -> io::Result<Vec<u32>> {
        let data = self.recv(n * 4)?;
        Ok(bytemuck::cast_slice::<u8, u32>(&data).to_vec())
    }

    #[cfg(not(target_endian = "little"))]
    fn recv_u32_slice(&mut self, n: usize) -> io::Result<Vec<u32>> {
        let data = self.recv(n * 4)?;
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 4;
            values.push(u32::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
            ]));
        }
        Ok(values)
    }

    /// Send a u64 value.
    fn send_u64(&mut self, v: u64) -> io::Result<()> {
        self.send(&v.to_le_bytes())
    }

    /// Receive a u64 value.
    fn recv_u64(&mut self) -> io::Result<u64> {
        let data = self.recv(8)?;
        Ok(u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]))
    }

    /// Send a slice of u64 values (zero-copy on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn send_u64_slice(&mut self, values: &[u64]) -> io::Result<()> {
        self.send(bytemuck::cast_slice(values))
    }

    #[cfg(not(target_endian = "little"))]
    fn send_u64_slice(&mut self, values: &[u64]) -> io::Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.send(&bytes)
    }

    /// Receive n u64 values (bulk cast on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn recv_u64_slice(&mut self, n: usize) -> io::Result<Vec<u64>> {
        let data = self.recv(n * 8)?;
        Ok(bytemuck::cast_slice::<u8, u64>(&data).to_vec())
    }

    #[cfg(not(target_endian = "little"))]
    fn recv_u64_slice(&mut self, n: usize) -> io::Result<Vec<u64>> {
        let data = self.recv(n * 8)?;
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 8;
            values.push(u64::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
            ]));
        }
        Ok(values)
    }

    /// Send a u128 value (as 16 bytes LE).
    fn send_u128(&mut self, v: u128) -> io::Result<()> {
        self.send(&v.to_le_bytes())
    }

    /// Receive a u128 value.
    fn recv_u128(&mut self) -> io::Result<u128> {
        let data = self.recv(16)?;
        Ok(u128::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]))
    }

    /// Send a slice of u128 values (zero-copy on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn send_u128_slice(&mut self, values: &[u128]) -> io::Result<()> {
        self.send(bytemuck::cast_slice(values))
    }

    #[cfg(not(target_endian = "little"))]
    fn send_u128_slice(&mut self, values: &[u128]) -> io::Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.send(&bytes)
    }

    /// Receive n u128 values (bulk cast on LE platforms via bytemuck).
    #[cfg(target_endian = "little")]
    fn recv_u128_slice(&mut self, n: usize) -> io::Result<Vec<u128>> {
        let data = self.recv(n * 16)?;
        Ok(bytemuck::cast_slice::<u8, u128>(&data).to_vec())
    }

    #[cfg(not(target_endian = "little"))]
    fn recv_u128_slice(&mut self, n: usize) -> io::Result<Vec<u128>> {
        let data = self.recv(n * 16)?;
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let offset = i * 16;
            values.push(u128::from_le_bytes([
                data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
                data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
                data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
                data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15],
            ]));
        }
        Ok(values)
    }
}

#[cfg(feature = "test-transport")]
mod memory_impl {
    use super::*;

    /// In-memory transport for testing, backed by crossbeam channels.
    pub struct MemoryTransport {
        tx: crossbeam::channel::Sender<Vec<u8>>,
        rx: crossbeam::channel::Receiver<Vec<u8>>,
        recv_buf: Vec<u8>,
    }

    impl Transport for MemoryTransport {
        fn send(&mut self, data: &[u8]) -> io::Result<()> {
            self.tx.send(data.to_vec()).map_err(|e| {
                io::Error::new(io::ErrorKind::BrokenPipe, e.to_string())
            })
        }

        fn recv(&mut self, len: usize) -> io::Result<Vec<u8>> {
            while self.recv_buf.len() < len {
                let data = self.rx.recv().map_err(|e| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, e.to_string())
                })?;
                self.recv_buf.extend_from_slice(&data);
            }
            Ok(self.recv_buf.drain(..len).collect())
        }

        fn recv_into(&mut self, buf: &mut [u8]) -> io::Result<()> {
            let len = buf.len();
            while self.recv_buf.len() < len {
                let data = self.rx.recv().map_err(|e| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, e.to_string())
                })?;
                self.recv_buf.extend_from_slice(&data);
            }
            buf.copy_from_slice(&self.recv_buf[..len]);
            self.recv_buf.drain(..len);
            Ok(())
        }
    }

    /// Create a pair of connected in-memory transports.
    ///
    /// Data sent on transport A is received on transport B, and vice versa.
    pub fn memory_transport_pair() -> (MemoryTransport, MemoryTransport) {
        let (tx_a, rx_a) = crossbeam::channel::unbounded();
        let (tx_b, rx_b) = crossbeam::channel::unbounded();

        let a = MemoryTransport { tx: tx_a, rx: rx_b, recv_buf: Vec::new() };
        let b = MemoryTransport { tx: tx_b, rx: rx_a, recv_buf: Vec::new() };

        (a, b)
    }
}

#[cfg(feature = "test-transport")]
pub use memory_impl::{MemoryTransport, memory_transport_pair};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_transport_send_recv() {
        let (mut a, mut b) = memory_transport_pair();

        a.send(&[1, 2, 3, 4]).unwrap();
        let received = b.recv(4).unwrap();
        assert_eq!(received, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_memory_transport_bidirectional() {
        let (mut a, mut b) = memory_transport_pair();

        a.send(&[10, 20]).unwrap();
        b.send(&[30, 40]).unwrap();

        let from_a = b.recv(2).unwrap();
        let from_b = a.recv(2).unwrap();

        assert_eq!(from_a, vec![10, 20]);
        assert_eq!(from_b, vec![30, 40]);
    }

    #[test]
    fn test_memory_transport_u32() {
        let (mut a, mut b) = memory_transport_pair();

        a.send_u32(0xDEADBEEF).unwrap();
        let v = b.recv_u32().unwrap();
        assert_eq!(v, 0xDEADBEEF);
    }

    #[test]
    fn test_memory_transport_u32_slice() {
        let (mut a, mut b) = memory_transport_pair();

        let values = vec![1u32, 2, 3, 0xFFFFFFFF];
        a.send_u32_slice(&values).unwrap();
        let received = b.recv_u32_slice(4).unwrap();
        assert_eq!(received, values);
    }

    #[test]
    fn test_memory_transport_partial_recv() {
        let (mut a, mut b) = memory_transport_pair();

        a.send(&[1, 2, 3]).unwrap();
        a.send(&[4, 5]).unwrap();

        // Receive more than one message's worth
        let received = b.recv(5).unwrap();
        assert_eq!(received, vec![1, 2, 3, 4, 5]);
    }
}
