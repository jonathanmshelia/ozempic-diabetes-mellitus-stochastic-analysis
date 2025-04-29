from typing import List
from scraper.Comment import Comment

class RawCommentParser:
    def __init__(self):
        pass

    def parse_comments(self, data: dict) -> List[Comment]:
        """Parse comments from Reddit API response"""
        self.comments = []
        
        # Reddit API returns two listings: post and comments
        if len(data) >= 2 and 'data' in data[1]:
            comments_data = data[1]['data']
            if 'children' in comments_data:
                for child in comments_data['children']:
                    if child['kind'] == 't1':  # t1 is comment type
                        comment_data = child['data']
                        comment = Comment.from_dict(comment_data)
                        self.comments.append(comment)
                        
                        # Parse replies if they exist
                        if 'replies' in comment_data and comment_data['replies']:
                            if isinstance(comment_data['replies'], dict) and 'data' in comment_data['replies']:
                                self._parse_replies(comment_data['replies']['data']['children'], comment)
        
        return self.comments
    
    def _parse_replies(self, replies_data, parent_comment) -> None:
        """Recursively parse nested replies"""
        for reply in replies_data:
            if reply['kind'] == 't1':
                reply_data = reply['data']
                reply_comment = Comment.from_dict(reply_data)
                parent_comment.children.append(reply_comment)
                
                # Recursively parse nested replies
                if 'replies' in reply_data and reply_data['replies']:
                    if isinstance(reply_data['replies'], dict) and 'data' in reply_data['replies']:
                        self._parse_replies(reply_data['replies']['data']['children'], reply_comment)
