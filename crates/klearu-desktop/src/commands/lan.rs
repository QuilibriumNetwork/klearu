//! LAN sharing commands — start/stop the embedded HTTP server that lets
//! browsers on the local network use Klearu without installing anything.

use std::sync::Arc;

use serde::Serialize;
use tauri::State;

use crate::error::{DesktopError, Result};
use crate::state::Workspace;

#[derive(Debug, Clone, Serialize)]
pub struct LanStatus {
    pub active: bool,
    pub port: Option<u16>,
    pub pin: Option<String>,
    /// LAN-accessible URLs constructed from every non-loopback IPv4 address
    /// on the machine. The user copies one and pastes it into the remote
    /// browser.
    pub urls: Vec<String>,
    /// SHA-256 fingerprint of the self-signed TLS cert. Shown on the
    /// desktop so the remote user can verify the cert matches when
    /// accepting it in their browser.
    pub cert_fingerprint: Option<String>,
}

#[tauri::command]
pub fn lan_status(workspace: State<'_, Arc<Workspace>>) -> LanStatus {
    let guard = workspace.lan_handle.lock();
    match guard.as_ref() {
        Some(h) => LanStatus {
            active: true,
            port: Some(h.port),
            pin: Some(h.pin.clone()),
            urls: local_urls(h.port),
            cert_fingerprint: Some(h.cert_fingerprint.clone()),
        },
        None => LanStatus { active: false, port: None, pin: None, urls: vec![], cert_fingerprint: None },
    }
}

#[tauri::command]
pub async fn lan_start(workspace: State<'_, Arc<Workspace>>, port: Option<u16>) -> Result<LanStatus> {
    // Stop any existing server first.
    {
        let mut guard = workspace.lan_handle.lock();
        if let Some(h) = guard.take() { h.stop(); }
    }
    let port = port.unwrap_or(3100);
    let ws: Arc<Workspace> = workspace.inner().clone();
    let handle = crate::lan_server::start(ws, port).await?;
    let status = LanStatus {
        active: true,
        port: Some(handle.port),
        pin: Some(handle.pin.clone()),
        urls: local_urls(handle.port),
        cert_fingerprint: Some(handle.cert_fingerprint.clone()),
    };
    *workspace.lan_handle.lock() = Some(handle);
    Ok(status)
}

#[tauri::command]
pub fn lan_stop(workspace: State<'_, Arc<Workspace>>) -> LanStatus {
    let mut guard = workspace.lan_handle.lock();
    if let Some(h) = guard.take() { h.stop(); }
    LanStatus { active: false, port: None, pin: None, urls: vec![], cert_fingerprint: None }
}

fn local_urls(port: u16) -> Vec<String> {
    let mut urls = Vec::new();
    if let Ok(addrs) = local_ipv4_addrs_pub() {
        for ip in addrs {
            urls.push(format!("https://{}:{}", ip, port));
        }
    }
    if urls.is_empty() {
        urls.push(format!("https://0.0.0.0:{port}"));
    }
    urls
}

pub fn local_ipv4_addrs_pub() -> std::io::Result<Vec<std::net::Ipv4Addr>> {
    let mut addrs = Vec::new();
    if let Ok(sock) = std::net::UdpSocket::bind("0.0.0.0:0") {
        if sock.connect("8.8.8.8:80").is_ok() {
            if let Ok(local) = sock.local_addr() {
                if let std::net::IpAddr::V4(ip) = local.ip() {
                    if !ip.is_loopback() {
                        addrs.push(ip);
                    }
                }
            }
        }
    }
    Ok(addrs)
}
