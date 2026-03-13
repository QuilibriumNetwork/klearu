//! TCP-based transport for 2PC protocols.
//!
//! Wraps a `TcpStream` with buffered I/O and implements the `Transport` trait.

use klearu_mpc::Transport;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::net::{TcpListener, TcpStream};

/// A transport backed by a TCP connection.
pub struct TcpTransport {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
}

impl TcpTransport {
    /// Create a new `TcpTransport` from a connected `TcpStream`.
    ///
    /// Sets TCP_NODELAY to avoid Nagle buffering, which is critical for the
    /// many small round-trips in the Beaver triple protocol.
    pub fn new(stream: TcpStream) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        let reader = BufReader::new(stream.try_clone()?);
        let writer = BufWriter::new(stream);
        Ok(Self { reader, writer })
    }
}

impl Transport for TcpTransport {
    fn send(&mut self, data: &[u8]) -> io::Result<()> {
        self.writer.write_all(data)?;
        self.writer.flush()
    }

    fn recv(&mut self, len: usize) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    fn recv_into(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf)
    }
}

/// Create a pair of connected TCP transports on localhost (for testing).
///
/// Binds an ephemeral port, connects, and returns `(client, server)`.
pub fn tcp_transport_pair() -> io::Result<(TcpTransport, TcpTransport)> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;

    let client_stream = TcpStream::connect(addr)?;
    let (server_stream, _) = listener.accept()?;

    let client = TcpTransport::new(client_stream)?;
    let server = TcpTransport::new(server_stream)?;
    Ok((client, server))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tcp_bidirectional() {
        let (mut a, mut b) = tcp_transport_pair().unwrap();

        a.send(&[1, 2, 3, 4]).unwrap();
        b.send(&[10, 20, 30, 40]).unwrap();

        let from_a = b.recv(4).unwrap();
        let from_b = a.recv(4).unwrap();

        assert_eq!(from_a, vec![1, 2, 3, 4]);
        assert_eq!(from_b, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_tcp_u32_slice_roundtrip() {
        let (mut a, mut b) = tcp_transport_pair().unwrap();

        let values = vec![42u32, 0xDEADBEEF, 0, u32::MAX];
        a.send_u32_slice(&values).unwrap();
        let received = b.recv_u32_slice(4).unwrap();
        assert_eq!(received, values);
    }

    #[test]
    fn test_tcp_large_transfer() {
        let (mut a, mut b) = tcp_transport_pair().unwrap();

        let data: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        a.send(&data).unwrap();
        let received = b.recv(10000).unwrap();
        assert_eq!(received, data);
    }
}
