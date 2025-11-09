#!/usr/bin/env python3
"""Query FDSN data centers and generate a markdown report of available data."""

from collections import defaultdict
from datetime import datetime, timedelta

from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException

# Configuration
TIMEOUT = 60
DAYS_BACK = 30
SAMPLE_NETWORKS = 3

DATA_CENTERS = [
    "IRIS",
    "GEOFON",
    "INGV",
    "ETH",
    "RESIF",
    "BGR",
    "GFZ",
    "USGS",
    "NCEDC",
    "SCEDC",
    "ORFEUS",
    "IPGP",
    "GEONET",
    "KNMI",
    "KOERI",
    "LMU",
    "NOA",
    "ISC",
]


def query_center(name):
    """Query a single data center for services, networks, and channel types."""
    result = {
        "name": name,
        "accessible": False,
        "services": [],
        "networks": [],
        "channels": set(),
        "error": None,
    }

    try:
        print(f"Querying {name}...")
        client = Client(name, timeout=TIMEOUT)
        result["accessible"] = True

        # Get services
        services = client.services
        if "dataselect" in services:
            result["services"].append("dataselect (waveforms)")
        if "station" in services:
            result["services"].append("station (metadata)")
        if "event" in services:
            result["services"].append("event (catalogs)")

        # Get networks
        endtime = datetime.now()
        starttime = endtime - timedelta(days=DAYS_BACK)

        inventory = client.get_stations(
            starttime=starttime,
            endtime=endtime,
            level="network",
            includerestricted=False,
        )

        result["networks"] = [
            {
                "code": net.code,
                "description": net.description or "N/A",
                "start": str(net.start_date) if net.start_date else "Unknown",
                "end": str(net.end_date) if net.end_date else "Ongoing",
            }
            for net in inventory
        ]

        # Sample channels from first few networks
        for net in inventory[:SAMPLE_NETWORKS]:
            try:
                detailed = client.get_stations(
                    network=net.code,
                    starttime=starttime,
                    endtime=endtime,
                    level="channel",
                    includerestricted=False,
                )
                for n in detailed:
                    for sta in n:
                        for ch in sta:
                            result["channels"].add(ch.code)
            except Exception:
                continue

        print(
            f"  ✓ {len(result['networks'])} networks, {len(result['channels'])} channel types"
        )

    except FDSNException as e:
        result["error"] = f"FDSN error: {str(e)}"
        print(f"  ✗ {result['error']}")
    except Exception as e:
        result["error"] = f"Connection error: {str(e)}"
        print(f"  ✗ {result['error']}")

    return result


def write_report(results, filename="fdsn_report.md"):
    """Write results to markdown file."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# FDSN Data Centers Report\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        # Summary
        accessible = sum(1 for r in results if r["accessible"])
        total_nets = sum(len(r["networks"]) for r in results)
        f.write(
            f"**Accessible:** {accessible}/{len(results)} centers, {total_nets} networks\n\n---\n\n"
        )

        # Details
        for r in results:
            f.write(f"## {r['name']}\n\n")

            if not r["accessible"]:
                f.write(f"❌ Not accessible: {r['error']}\n\n---\n\n")
                continue

            f.write("✅ Accessible\n\n")

            # Services
            if r["services"]:
                f.write("**Services:** " + ", ".join(r["services"]) + "\n\n")

            # Networks table
            if r["networks"]:
                f.write(f"**Networks:** {len(r['networks'])}\n\n")
                f.write(
                    "| Code | Description | Start | End |\n|------|-------------|-------|-----|\n"
                )
                for net in r["networks"][:15]:
                    desc = (
                        net["description"][:40] + "..."
                        if len(net["description"]) > 40
                        else net["description"]
                    )
                    f.write(
                        f"| {net['code']} | {desc} | {net['start'][:10]} | {net['end'][:10]} |\n"
                    )
                if len(r["networks"]) > 15:
                    f.write(f"\n*...and {len(r['networks']) - 15} more*\n")
                f.write("\n")

            # Channels
            if r["channels"]:
                f.write(f"**Channel Types:** {', '.join(sorted(r['channels']))}\n\n")

            f.write("---\n\n")

    print(f"\n✓ Report: {filename}")


def main():
    """Main execution."""
    print(
        f"Querying {len(DATA_CENTERS)} FDSN centers (timeout={TIMEOUT}s, window={DAYS_BACK}d)\n"
    )

    results = []
    for i, center in enumerate(DATA_CENTERS, 1):
        print(f"[{i}/{len(DATA_CENTERS)}] ", end="")
        results.append(query_center(center))

    write_report(results)
    print("✓ Done!")


if __name__ == "__main__":
    main()
