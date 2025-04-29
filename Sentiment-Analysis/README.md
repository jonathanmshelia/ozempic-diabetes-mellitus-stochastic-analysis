# sentiment-analyzer
Custom utility library to parse social-media contents

## Notebooks

- `comment-accumulator.ipynb`: Notebook for accumulating comments from the subreddit_posts.txt file.
- `comment-collector.ipynb`: Notebook for collecting comments.
- `example-scraper.ipynb`: Example notebook demonstrating how to scrape comments.

## Dataset

- `comments/`: Contains raw and cleaned comment CSV files.
  - `cleaned_comments.csv`: Cleaned comments data.
  - `comments.csv`: Raw comments data.
- `output/`: Contains JSON files with processed comments.
- `sample/`: Contains sample CSV and JSON files with comments.

## Scraper

- `scraper/__init__.py`: Initialization file for the scraper module.
- `scraper/Comment.py`: Defines the `Comment` class.
- `scraper/JsonLoader.py`: Loads JSON data.
- `scraper/RawCommentParser.py`: Parses raw comments.
- `scraper/RedditCommentLoader.py`: Loads comments from Reddit.
- `scraper/RedditCommentSerializer.py`: Serializes Reddit comments.
- `scraper/RedditJsonLoader.py`: Loads Reddit JSON data.

## Requirements

See `requirements.txt` for the list of dependencies.

## Usage

1. Clone the repository.
2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the notebooks or scripts as needed.

## License

This project is licensed under the MIT License.

