import csv
import json
import pandas as pd
from typing import Dict, List
from scraper.Comment import Comment

class RedditCommentSerializer:
    def __init__(self):
        pass
    
    def write_to_csv(self, comments: List[Comment], path: str) -> None:
        """Write comments to a CSV file"""
        flat_comments = self._flatten_comments(comments)
        fieldnames = Comment.__dataclass_fields__.keys()
        fieldnames = [field if field != '_id' else 'id' for field in fieldnames] # Replace _id with id for CSV header
        fieldnames.remove('children') # Remove children field from CSV
        with open(path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for comment in flat_comments:
                writer.writerow(comment)

    def append_to_csv(self, comments: List[Comment], path: str, header: bool = False) -> None:
        """Append comments to a CSV file"""
        flat_comments = self._flatten_comments(comments)
        fieldnames = Comment.__dataclass_fields__.keys()
        fieldnames = [field if field != '_id' else 'id' for field in fieldnames]
        fieldnames.remove('children')
        with open(path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if header:
                writer.writeheader()
            for comment in flat_comments:
                writer.writerow(comment)

    def write_to_json(self, comments: List[Comment], path: str) -> None:
        """Write comments to a JSON file"""
        flat_comments = self._flatten_comments(comments)
        
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(flat_comments, file, indent=4)
        
    def _flatten_comments(self, comments: List[Comment], parent_id=None, depth=0) -> List[Dict]:
        """Convert nested comment structure to flat list for serialization"""
        flat_list = []
        
        for comment in comments:
            comment_dict = comment.to_dict()
            flat_list.append(comment_dict)
            
            # Add children recursively
            if comment.children:
                flat_list.extend(self._flatten_comments(
                    comment.children, 
                    parent_id=comment._id,
                    depth=depth+1
                ))
        
        return flat_list
    
    def to_dataframe(self, comments: List[Comment]):
        """Convert comments to pandas DataFrame"""
        flat_comments = self._flatten_comments(comments)
        return pd.DataFrame(flat_comments)
