from typing import List
from scraper.Comment import Comment
from scraper.JsonLoader import JsonLoader

class RedditJsonLoader(JsonLoader):
    def __init__(self):
        super().__init__()
        pass
    
    def load_comments(self, path) -> List[Comment]:
        raw = self.load_raw(path)
        if raw is None:
            return None
        return [Comment.from_dict(comment) for comment in raw]