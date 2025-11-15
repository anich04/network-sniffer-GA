#!/usr/bin/env python3
import argparse
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP

def pkt_to_features(pkt):
    row = {
        "src_ip": None, "dst_ip": None,
        "src_port": None, "dst_port": None,
        "protocol": None, "size": len(pkt),
        "ttl": None, "flags": None,
        "timestamp": float(getattr(pkt, "time", 0.0))
    }
    if IP in pkt:
        row["src_ip"] = pkt[IP].src
        row["dst_ip"] = pkt[IP].dst
        row["ttl"] = pkt[IP].ttl
        row["protocol"] = pkt[IP].proto
    if TCP in pkt:
        row["src_port"] = pkt[TCP].sport
        row["dst_port"] = pkt[TCP].dport
        row["flags"] = str(pkt[TCP].flags)
        row["protocol"] = "TCP"
    elif UDP in pkt:
        row["src_port"] = pkt[UDP].sport
        row["dst_port"] = pkt[UDP].dport
        row["protocol"] = "UDP"
    return row

def extract(pcap, out_csv):
    pkts = rdpcap(pcap)
    rows = [pkt_to_features(p) for p in pkts]
    df = pd.DataFrame(rows)
    df["time_diff"] = df["timestamp"].diff().fillna(0)
    df.to_csv(out_csv, index=False)
    print(f"Saved features to {out_csv}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcap", default="src/data/raw/sample.pcap")
    parser.add_argument("--out", default="src/data/processed/features.csv")
    args = parser.parse_args()
    extract(args.pcap, args.out)

if __name__ == "__main__":
    main()
