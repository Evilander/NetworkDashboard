[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Streamlit-based dashboard for real-time network traffic monitoring, analysis, and visualization.

## Features

*   **Real-time Packet Capture:** Captures network traffic using Scapy.
*   **Dashboard Overview:** Displays key metrics like total packets, capture duration, average packet size, and unique protocols.
*   **Visualizations:**
    *   Protocol Distribution Pie Chart
    *   Packet Timeline Line Chart
    *   Top Talkers (Source/Destination IPs) Bar Charts & Tables
*   **Network Graph:** Visualizes connections between top talkers using NetworkX and Plotly.
*   **Trend Analysis:** Analyzes traffic volume trends over selected periods (hour, day, week, month).
*   **Security Insights (Optional LLM):** Provides basic traffic analysis, anomaly detection, and security recommendations. Can leverage an LLM (like GPT-4) for deeper insights if configured.
*   **Data Filtering:** Filter displayed data by timeframe and protocol.
*   **Data Management:** Options to optimize the database, clean up old data, and reset the dashboard.
*   **Memory Management:** Monitors RAM usage and performs cleanup actions to prevent crashes.
*   **Configurable:** Settings managed via environment variables (`.env` file).


## Requirements

*   Python 3.8+
*   `libpcap` or `Npcap`: Scapy's underlying packet capture library.
    *   **Linux:** `sudo apt update && sudo apt install libpcap-dev` or `sudo yum install libpcap-devel`
    *   **macOS:** Usually included, or install via Homebrew: `brew install libpcap`
    *   **Windows:** Install [Npcap](https://npcap.com/#download) (select "Install Npcap in WinPcap API-compatible Mode" during installation).
*   Required Python packages are listed in `requirements.txt`.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Evilander/NetworkDashboard/ 
    cd NetworkDashboard
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate it:
    # Linux/macOS:
    source venv/bin/activate
    # Windows:
    .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file with your settings:
        *   **`LLM_API_KEY` (Optional):** Add your API key if you want to use the LLM-based security insights.
        *   **`LLM_API_ENDPOINT` & `LLM_MODEL` (Optional):** Adjust if using a different LLM provider or model.
        *   Other settings (DB path, memory limits, etc.) can be adjusted if needed, otherwise defaults will be used.
    *   **IMPORTANT:** Never commit your `.env` file to version control. It's included in `.gitignore`.

## Running the Dashboard

*   **Permissions:** Packet capture requires elevated privileges.
    *   **Linux/macOS:** Run using `sudo`:
        ```bash
        sudo venv/bin/python dashboard.py
        ```
        *(Adjust `venv/bin/python` if your virtual environment path differs)*
    *   **Windows:** Run your terminal (Command Prompt or PowerShell) as Administrator, then run:
              streamlit run dashboard.py 

*   Once running, open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Configuration

Most configuration is handled via the `.env` file (see `Installation` step 4 and `.env.example`). Key options include:

*   `LLM_API_KEY`, `LLM_API_ENDPOINT`, `LLM_MODEL`: For LLM integration.
*   `DB_PATH`: Location to store the traffic database.
*   `MEMORY_WARNING_THRESHOLD`, `MEMORY_CRITICAL_THRESHOLD`: RAM usage thresholds for warnings and cleanup.
*   `AUTO_CLEANUP_DAYS`: How long to retain packet data.
*   `MAX_PACKETS_STORED`: Limit on total packets stored (approximate, cleanup runs periodically).

## Security Considerations

*   **Running with Root/Admin:** This application requires elevated privileges for packet capture, which inherently carries security risks. Run it only in trusted environments.
*   **Data Privacy:** Network traffic can contain sensitive information. Ensure you have the necessary permissions to capture and analyze traffic on your network and comply with relevant privacy regulations. The captured data is stored locally in the SQLite database (`network_traffic.db` by default).
*   **LLM API Key:** Keep your LLM API key secure in the `.env` file and do not share it or commit it to Git.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## Disclaimer

This tool is for educational and informational purposes. Use responsibly and ethically. The developers assume no liability for misuse or damages caused by this software.
