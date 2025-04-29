import json
import requests
from typing import Dict, List

class RedditCommentLoader:
    def __init__(self):
        pass

    def _clean_url(self, url: str) -> str:
        # Add .json to the URL if not already present
        if not url.endswith('.json'):
            if url.endswith('/'):
                url = url + '.json'
            else:
                url = url + '/.json'
        return url

    def get_comments(self, url) -> List[Dict]:
        url = self._clean_url(url)
        # Set a user agent to avoid being blocked
        headers = {
            'User-Agent': 'python:reddit-comment-loader:v1.0 (by /u/yourname)'
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching comments: HTTP {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None