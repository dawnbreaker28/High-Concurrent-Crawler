# High Concurrent Crawler

## Description

High Concurrent Crawler is a Python-based web crawler designed for high concurrency and efficiency. It reads URLs from a source file and performs breadth-first search (BFS) with a depth of 3. The crawler uses the HITS algorithm to classify web pages as hubs or authorities. Content from authority pages is then fetched and saved for further analysis.

## Features

- **High Concurrency**: Utilizes Python's `concurrent.futures` module to send HTTP requests concurrently.
- **Breadth-First Search**: Implements BFS to crawl web pages up to a specified depth.
- **HITS Algorithm**: Determines hub and authority scores for each web page.
- **Content Extraction**: Fetches and saves content from authority pages.
- **Modular Design**: Clean and organized code structure with separate modules for fetching, parsing, BFS, and HITS.


## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/high_concurrent_crawler.git
    cd high_concurrent_crawler
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Configure the source file and other settings** in `config.py`.

2. **Run the crawler**:
    ```bash
    python main.py
    ```

## Contributions

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

## License

This project is licensed under the MIT License.

