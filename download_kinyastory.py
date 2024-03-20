import requests
import os
import json
from pathlib import Path
import pandas as pd
import json

class DownloadKinyastory:
    def __init__(self):
        self.url = "https://storytelling-m5a9.onrender.com/stories"
        self.save_dir = 'kinyastory_data'
        self.save_path = os.path.join(os.getcwd(), self.save_dir)
        self.file_path = os.path.join(self.save_path, "stories.json")

        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} stories from {self.file_path}")
        else:
            response = requests.get(self.url)
            if response.status_code == 200:
                self.data = response.json()
            else:
                raise Exception(f"Failed to fetch data from {self.url}. Status code: {response.status_code}")
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __len__(self):
        return len(self.data)
    
    def download(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump(self.data, f)
            print(f"Downloaded {len(self.data)} stories to {self.file_path}")



class MakeUsableDataset:
    def __init__(self, stories,directory):
        self.stories = stories
        self.save_path = os.path.join(os.getcwd(),directory)

    def create_csv(self, filename):
        data = []
        for story in self.stories:
            demographic_values = ''
            if story['demographic']:
                try:
                    demographic_dict = json.loads(story['demographic'])
                    if isinstance(demographic_dict, dict):
                        demographic_values = ' '.join(map(str, demographic_dict.values()))
                except json.JSONDecodeError:
                    demographic_values = story['demographic']
            story_input = f"{story['story_title']} {story['genre']} {story['size']} {demographic_values} {story['themes']}"
            data.append({
                'storyId': story['story_id'],
                'story_input': story_input,
                'story_output': story['story_text']
            })
        df = pd.DataFrame(data)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        df.to_csv(os.path.join(self.save_path, filename), index=False)
    def _len_(self):
        return len(self.stories)
    
if __name__ == "__main__":
    kinyastory = DownloadKinyastory()
    kinyastory.download()
    stories = kinyastory.data
    dataset = MakeUsableDataset(stories, 'kinyastory_data')
    dataset.create_csv('kinyastory.csv')
    print(f"Created kinyastory.csv with {len(stories)} stories")
    print(f"Saved in {os.getcwd()}")
   

