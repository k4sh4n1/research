#!/usr/bin/env python3
"""
Optimized script to query FDSN data centers and catalog their available data types.
Requires: obspy
Install with: pip install obspy
"""

import datetime
from collections import defaultdict

from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNException


def get_fdsn_data_centers():
    """
    Returns a list of major FDSN data centers.
    """
    return [
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


def query_data_center(data_center_name, timeout=30, days_back=30):
    """
    Query a single data center for available services and data types.

    Parameters:
    - data_center_name: Name of the FDSN data center
    - timeout: Timeout in seconds for queries (default: 30)
    - days_back: How many days back to query (default: 30 for faster queries)
    """
    result = {
        "name": data_center_name,
        "accessible": False,
        "services": [],
        "networks": [],
        "channel_types": set(),
        "error": None,
    }

    try:
        print(f"Querying {data_center_name}...")
        client = Client(data_center_name, timeout=timeout)
        result["accessible"] = True

        # Get available services
        services = client.services
        if "dataselect" in services:
            result["services"].append("dataselect (waveform data)")
        if "station" in services:
            result["services"].append("station (metadata)")
        if "event" in services:
            result["services"].append("event (earthquake catalogs)")

        print(f"  Services found: {len(result['services'])}")

        # Try to get a LIMITED sample of available networks
        try:
            # Query for recent networks (reduced time window)
            endtime = datetime.datetime.now()
            starttime = endtime - datetime.timedelta(days=days_back)

            print(
                f"  Fetching network metadata (last {days_back} days, this may take a moment)..."
            )

            # Use a more limited query - get only network level first
            inventory = client.get_stations(
                starttime=starttime,
                endtime=endtime,
                level="network",  # Changed from 'channel' to 'network' for speed
                includerestricted=False,
            )

            print(f"  Found {len(inventory)} networks, now getting sample channels...")

            # Extract network codes
            for network in inventory:
                result["networks"].append(
                    {
                        "code": network.code,
                        "description": network.description or "N/A",
                        "start_date": str(network.start_date)
                        if network.start_date
                        else "Unknown",
                        "end_date": str(network.end_date)
                        if network.end_date
                        else "Ongoing",
                    }
                )

            # Get channel types from a LIMITED sample of networks
            sample_size = min(3, len(inventory))  # Sample only 3 networks
            print(f"  Sampling {sample_size} networks for channel types...")

            for i, network in enumerate(inventory[:sample_size]):
                try:
                    # Get detailed info for just this network
                    detailed_inv = client.get_stations(
                        network=network.code,
                        starttime=starttime,
                        endtime=endtime,
                        level="channel",
                        includerestricted=False,
                    )

                    for net in detailed_inv:
                        for station in net:
                            for channel in station:
                                result["channel_types"].add(channel.code)

                    print(f"    Sampled network {i + 1}/{sample_size}: {network.code}")
                except Exception as e:
                    print(
                        f"    Warning: Could not sample network {network.code}: {str(e)}"
                    )
                    continue

            print(
                f"  ✓ Complete: {len(result['networks'])} networks, {len(result['channel_types'])} channel types"
            )

        except FDSNException as e:
            result["error"] = f"Could not retrieve network details: {str(e)}"
            print(f"  Warning: {result['error']}")
        except Exception as e:
            result["error"] = f"Unexpected error retrieving networks: {str(e)}"
            print(f"  Warning: {result['error']}")

    except FDSNException as e:
        result["error"] = f"FDSN Error: {str(e)}"
        print(f"  ✗ Error: Cannot connect to {data_center_name}")
    except Exception as e:
        result["error"] = f"Connection error: {str(e)}"
        print(f"  ✗ Error: {result['error']}")

    return result


def write_markdown_report(results, filename="fdsn_data_centers_report.md"):
    """
    Write the results to a markdown file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# FDSN Data Centers and Available Data Types\n\n")
        f.write(
            f"*Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        )
        f.write(
            "This report catalogs available FDSN data centers and the types of seismological data they provide.\n\n"
        )
        f.write(
            "*Note: Channel types are sampled from a subset of networks for performance reasons.*\n\n"
        )

        # Summary statistics
        accessible_centers = sum(1 for r in results if r["accessible"])
        total_networks = sum(len(r["networks"]) for r in results)

        f.write("## Summary\n\n")
        f.write(f"- **Total Data Centers Queried:** {len(results)}\n")
        f.write(f"- **Accessible Data Centers:** {accessible_centers}\n")
        f.write(f"- **Total Networks Found:** {total_networks}\n\n")

        f.write("---\n\n")

        # Detailed information for each data center
        for result in results:
            f.write(f"## {result['name']}\n\n")

            if not result["accessible"]:
                f.write(f"**Status:** ❌ Not accessible\n\n")
                if result["error"]:
                    f.write(f"**Error:** {result['error']}\n\n")
                f.write("---\n\n")
                continue

            f.write(f"**Status:** ✅ Accessible\n\n")

            # Services
            if result["services"]:
                f.write("### Available Services\n\n")
                for service in result["services"]:
                    f.write(f"- {service}\n")
                f.write("\n")

            # Networks
            if result["networks"]:
                f.write(f"### Networks ({len(result['networks'])} found)\n\n")
                f.write("| Network Code | Description | Start Date | End Date |\n")
                f.write("|--------------|-------------|------------|----------|\n")
                for network in result["networks"][:20]:
                    desc = (
                        network["description"][:50] + "..."
                        if len(network["description"]) > 50
                        else network["description"]
                    )
                    f.write(
                        f"| {network['code']} | {desc} | {network['start_date'][:10]} | {network['end_date'][:10] if network['end_date'] != 'Ongoing' else 'Ongoing'} |\n"
                    )

                if len(result["networks"]) > 20:
                    f.write(
                        f"\n*... and {len(result['networks']) - 20} more networks*\n"
                    )
                f.write("\n")

            # Channel types
            if result["channel_types"]:
                f.write("### Channel Types (Sampled)\n\n")
                f.write(
                    "Available channel codes (format: Band-Instrument-Orientation):\n\n"
                )

                # Group by band code
                channel_dict = defaultdict(list)
                for ch in sorted(result["channel_types"]):
                    if len(ch) >= 3:
                        band = ch[0]
                        channel_dict[band].append(ch)

                for band, channels in sorted(channel_dict.items()):
                    f.write(f"- **{band}-band:** {', '.join(sorted(channels))}\n")

                f.write("\n")
                f.write("<details>\n<summary>Channel Code Reference</summary>\n\n")
                f.write("**Band Codes:**\n")
                f.write("- B = Broadband (10-80 Hz)\n")
                f.write("- H = High Broadband (≥80 Hz)\n")
                f.write("- L = Long Period\n")
                f.write("- V = Very Long Period\n")
                f.write("- E = Extremely Short Period\n")
                f.write("- S = Short Period\n\n")
                f.write("**Instrument Codes:**\n")
                f.write("- H = High Gain Seismometer\n")
                f.write("- L = Low Gain Seismometer\n")
                f.write("- N = Accelerometer\n\n")
                f.write("**Orientation Codes:**\n")
                f.write("- Z = Vertical\n")
                f.write("- N = North\n")
                f.write("- E = East\n")
                f.write("- 1,2,3 = Orthogonal components\n")
                f.write("</details>\n\n")

            if result["error"]:
                f.write(f"**Note:** {result['error']}\n\n")

            f.write("---\n\n")

    print(f"\n✓ Report written to: {filename}")


def main():
    """
    Main function to query all data centers and generate report.
    """
    print("=" * 60)
    print("FDSN Data Center Query Script (Optimized)")
    print("=" * 60)
    print()

    data_centers = get_fdsn_data_centers()
    results = []

    # Configuration
    TIMEOUT = 60  # Increase timeout to 60 seconds
    DAYS_BACK = 30  # Query only last 30 days (instead of 365)

    print(f"Configuration:")
    print(f"  - Timeout: {TIMEOUT} seconds")
    print(f"  - Time window: Last {DAYS_BACK} days")
    print(f"  - Channel sampling: 3 networks per data center")
    print()
    print(f"Querying {len(data_centers)} FDSN data centers...")
    print("(This may take 5-15 minutes depending on data center response times)")
    print()

    for i, dc in enumerate(data_centers, 1):
        print(f"[{i}/{len(data_centers)}] ", end="")
        result = query_data_center(dc, timeout=TIMEOUT, days_back=DAYS_BACK)
        results.append(result)
        print()

    print("=" * 60)
    print("Query complete! Generating markdown report...")
    print("=" * 60)

    write_markdown_report(results)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
