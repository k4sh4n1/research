#!/usr/bin/env python3
"""
Script to query FDSN data centers and catalog their available data types.
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
    Based on the FDSN registry.
    """
    return [
        "IRIS",  # IRIS Data Management Center (USA)
        "GEOFON",  # GEOFON Program (Germany)
        "INGV",  # INGV Data Centre (Italy)
        "ETH",  # ETH Data Centre (Switzerland)
        "RESIF",  # RESIF Data Center (France)
        "BGR",  # BGR Data Centre (Germany)
        "GFZ",  # GeoForschungsZentrum Potsdam (Germany)
        "USGS",  # USGS (USA)
        "NCEDC",  # Northern California Earthquake Data Center
        "SCEDC",  # Southern California Earthquake Data Center
        "ORFEUS",  # ORFEUS Data Center (Europe)
        "IPGP",  # IPGP Data Center (France)
        "GEONET",  # GeoNet (New Zealand)
        "KNMI",  # Royal Netherlands Meteorological Institute
        "KOERI",  # Bogazici University Kandilli Observatory (Turkey)
        "LMU",  # Ludwig Maximilian University of Munich
        "NOA",  # National Observatory of Athens (Greece)
        "ISC",  # International Seismological Centre (UK)
    ]


def query_data_center(data_center_name):
    """
    Query a single data center for available services and data types.
    Returns a dictionary with information about the data center.
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
        client = Client(data_center_name)
        result["accessible"] = True

        # Get available services
        services = client.services
        if "dataselect" in services:
            result["services"].append("dataselect (waveform data)")
        if "station" in services:
            result["services"].append("station (metadata)")
        if "event" in services:
            result["services"].append("event (earthquake catalogs)")

        # Try to get a sample of available networks (limited query)
        try:
            # Query for recent networks (last year)
            endtime = datetime.datetime.now()
            starttime = endtime - datetime.timedelta(days=365)

            inventory = client.get_stations(
                starttime=starttime,
                endtime=endtime,
                level="channel",
                includerestricted=False,
            )

            # Extract network codes and channel types
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

                # Get channel types from stations
                for station in network:
                    for channel in station:
                        # Channel code format: BHZ, HHZ, etc.
                        # First letter: Band code (B=broadband, H=high broadband, etc.)
                        # Second letter: Instrument code (H=high gain, L=low gain, etc.)
                        # Third letter: Orientation (Z=vertical, N=north, E=east, etc.)
                        result["channel_types"].add(channel.code)

            print(
                f"  Found {len(result['networks'])} networks with {len(result['channel_types'])} channel types"
            )

        except FDSNException as e:
            result["error"] = f"Could not retrieve network details: {str(e)}"
            print(f"  Warning: {result['error']}")
        except Exception as e:
            result["error"] = f"Unexpected error retrieving networks: {str(e)}"
            print(f"  Warning: {result['error']}")

    except FDSNException as e:
        result["error"] = f"FDSN Error: {str(e)}"
        print(f"  Error: Cannot connect to {data_center_name}")
    except Exception as e:
        result["error"] = f"Connection error: {str(e)}"
        print(f"  Error: {result['error']}")

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
                for network in result["networks"][
                    :20
                ]:  # Limit to first 20 for readability
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
                f.write("### Channel Types\n\n")
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

    print(f"\nReport written to: {filename}")


def main():
    """
    Main function to query all data centers and generate report.
    """
    print("=" * 60)
    print("FDSN Data Center Query Script")
    print("=" * 60)
    print()

    data_centers = get_fdsn_data_centers()
    results = []

    print(f"Querying {len(data_centers)} FDSN data centers...\n")

    for dc in data_centers:
        result = query_data_center(dc)
        results.append(result)
        print()

    print("=" * 60)
    print("Query complete! Generating markdown report...")
    print("=" * 60)

    write_markdown_report(results)

    print("\nDone!")


if __name__ == "__main__":
    main()
