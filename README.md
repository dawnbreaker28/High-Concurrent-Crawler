# High-Concurrent-Crawler
High Concurrent Crawler is a Python-based web crawler designed for high concurrency and efficiency. It reads URLs from a source file and performs breadth-first search (BFS) with a depth of 3. The crawler uses the HITS algorithm to classify web pages as hubs or authorities. Content from authority pages is then fetched and saved for further analysis.

Features
High Concurrency: Utilizes Python's concurrent.futures module to send HTTP requests concurrently.
Breadth-First Search: Implements BFS to crawl web pages up to a specified depth.
HITS Algorithm: Determines hub and authority scores for each web page.
Content Extraction: Fetches and saves content from authority pages.
Modular Design: Clean and organized code structure with separate modules for fetching, parsing, BFS, and HITS.
Project Structure
bash
Copy code
high_concurrent_crawler/
├── main.py               # Main entry point of the application
├── config.py             # Configuration settings
├── crawler/              # Crawler related modules
│   ├── __init__.py
│   ├── fetcher.py        # HTTP request handling
│   ├── parser.py         # HTML content parsing
│   ├── bfs.py            # Breadth-First Search implementation
│   └── hits.py           # HITS algorithm implementation
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── file_io.py        # File input/output handling
│   └── logger.py         # Logging setup
├── results/
│   └── authority_content/  # Directory to save authority page content
└── requirements.txt      # Project dependencies
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/high_concurrent_crawler.git
cd high_concurrent_crawler
Create and activate a virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Configure the source file and other settings in config.py.

Run the crawler:

bash
Copy code
python main.py
Contributions
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

License
This project is licensed under the MIT License.


