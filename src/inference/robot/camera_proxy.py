#!/usr/bin/env python3
"""
camera_proxy.py — ZMQ camera frame relay (Orin → inference server).

Run this ON THE ORIN next to deploy_observer.py.
Subscribe to localhost:5549 (deploy_observer's commlink Publisher), stamp proxy
timing, republish on 0.0.0.0:5550. --pub-host defaults to 0.0.0.0 (not 127.0.0.1
-- see 2026-09 history) because this deployment reaches the Orin directly over the
robot LAN (192.168.123.x), not through the SSH reverse-tunnel-to-a-remote-Maxwell
topology the original docstring below describes. A loopback-only bind here is a
silent failure mode: the inference server's Camera SUB (default 192.168.123.116:5550)
just never receives anything, and the only sign is a WARNING buried in that
process's own log -- it cost real debugging time more than once before this
default was fixed. If you genuinely need the old remote-tunnel topology, pass
--pub-host 127.0.0.1 explicitly and set up the tunnel below yourself.
  ssh -N -R 5550:127.0.0.1:5550 aap9348@128.238.176.100

Wire contract: this uses commlink's own Subscriber/Publisher on both hops.
commlink's Publisher is a zmq.ROUTER (NOT a plain PUB) that speaks a custom
OP_SUB / OP_PUSH protocol over a DEALER, so the only thing that can receive from
it — or feed a downstream commlink Subscriber — is commlink itself. A raw
zmq.SUB/PUB pair is silently incompatible and receives nothing.

  IN  topic "obs/cameras" on :5549  (push mode: every frame delivered)
  OUT topic "cameras"     on :5550
The proxy adds pkt["_proxy"] with recv/send timing before republishing.
"""

import argparse
import logging
import time

from commlink import Publisher, Subscriber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("camera_proxy")

_CAM_TOPIC_IN  = "obs/cameras"
_CAM_TOPIC_OUT = "cameras"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orin-host", default="127.0.0.1")
    parser.add_argument("--orin-port", type=int, default=5549)
    parser.add_argument("--pub-host",  default="0.0.0.0")
    parser.add_argument("--pub-port",  type=int, default=5550)
    parser.add_argument("--log-every", type=float, default=5.0)
    args = parser.parse_args()

    # Push mode (buffer=True): every published frame is delivered into a bounded
    # per-topic queue; on overflow the oldest is dropped, so get() stays close to
    # the newest frame and latency can't accumulate. Small queue = tight latency.
    sub = Subscriber(
        args.orin_host, port=args.orin_port,
        topics=[_CAM_TOPIC_IN], buffer=True, queue_size=2,
    )
    log.info("SUB  → tcp://%s:%d  topic=%r", args.orin_host, args.orin_port, _CAM_TOPIC_IN)

    pub = Publisher(args.pub_host, port=args.pub_port, queue_size=2)
    log.info("PUB  ← tcp://%s:%d  topic=%r  (LAN-reachable; pass --pub-host 127.0.0.1 for the "
             "old remote-tunnel-to-Maxwell topology instead -- see module docstring)",
             args.pub_host, args.pub_port, _CAM_TOPIC_OUT)

    frame_count = 0
    fps_count   = 0
    t_window    = time.monotonic()
    last_recv_t = 0.0
    rolling_fps = 0.0

    log.info("Proxy running — waiting for first camera frame from deploy_observer...")

    while True:
        # Blocks until the next frame on obs/cameras arrives.
        pkt = sub.get(_CAM_TOPIC_IN)
        recv_t = time.monotonic()

        if last_recv_t == 0.0:
            log.info("First frame received from deploy_observer.")

        pkt["_proxy"] = {
            "recv_t":    recv_t,
            "send_t":    None,      # filled just before send
            "frame_idx": frame_count,
            "fps":       rolling_fps,
        }
        send_t = time.monotonic()
        pkt["_proxy"]["send_t"] = send_t

        pub[_CAM_TOPIC_OUT] = pkt

        frame_count += 1
        fps_count   += 1
        last_recv_t  = recv_t

        elapsed = recv_t - t_window
        if elapsed >= args.log_every:
            rolling_fps = fps_count / elapsed
            proxy_lat   = (send_t - recv_t) * 1000
            log.info("fps=%.1f  frames=%d  proxy_lat=%.2fms",
                     rolling_fps, frame_count, proxy_lat)
            fps_count = 0
            t_window  = recv_t


if __name__ == "__main__":
    main()
