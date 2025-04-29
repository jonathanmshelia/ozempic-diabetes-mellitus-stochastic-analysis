import json
import os
from typing import Dict, List

class JsonLoader:
    def __init__(self):
        pass

    def load_raw(self, path) -> List[Dict]:
        if not os.path.exists(path):
            return None
        with open(path, 'r') as file:
            return json.load(file)