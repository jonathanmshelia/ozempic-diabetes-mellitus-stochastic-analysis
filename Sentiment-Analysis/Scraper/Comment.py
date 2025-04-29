from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Comment:
    """ Dataclass for a Reddit comment """
    _id: str
    author: str
    body: str
    created_utc: int
    score: int
    parent_id: str | None
    depth: int
    ups: int
    downs: int
    num_reports: int | None
    report_reasons: str | None
    children: List['Comment']

    def to_dict(self) -> Dict:
        return {
            'id': self._id,
            'author': self.author,
            'body': self.body,
            'created_utc': self.created_utc,
            'score': self.score,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'ups': self.ups,
            'downs': self.downs,
            'num_reports': self.num_reports,
            'report_reasons': self.report_reasons,
            # 'children': [c.to_dict() for c in self.children] # Omitted because the serializers handle this recursively
        }

    @classmethod
    def from_dict(cls, dict: Dict) -> 'Comment':
        return cls(
            _id=dict['id'],
            author=dict['author'],
            body=dict['body'],
            created_utc=dict['created_utc'],
            score=dict['score'],
            parent_id=dict['parent_id'],
            depth=dict['depth'],
            ups=dict['ups'],
            downs=dict['downs'],
            num_reports=dict['num_reports'],
            report_reasons=dict['report_reasons'],
            children=[cls.from_dict(c) for c in dict.get('children', [])]
        )