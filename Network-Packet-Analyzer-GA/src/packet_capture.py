#!/usr/bin/env python3
import argparse
from scapy.all import rdpcap, sniff, wrpcap

def read_pcap(path):
    print(f"Reading pcap: {path}")
    pkts = rdpcap(path)
    print(f"Loaded {len(pkts)} packets")
    return pkts

def sniff_live(count=10, iface=None, out_path="src/data/raw/live_capture.pcap"):
    print(f"Sniffing {count} packets...")
    pkts = sniff(count=count, iface=iface)
    wrpcap(out_path, pkts)
    print(f"Saved live capture to {out_path}")
    return pkts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", help="Path to .pcap file")
    parser.add_argument("--sniff", action="store_true")
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    if args.pcap:
        read_pcap(args.pcap)
    elif args.sniff:
        sniff_live(count=args.count)
    else:
        print("Use --pcap <file> or --sniff")

if __name__ == "__main__":
    main()
