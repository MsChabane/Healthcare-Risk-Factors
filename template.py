import os 
from pathlib import  Path



files = [
    "data/raw/.gitkeep",
    "data/cleaned/.gitkeep",
    "data/processed/.gitkeep",
    'notebook/.gitkeep',
    f'src/__init__.py',
    f'src/utils/.gitkeep',
    f'src/constant/.gitkeep',
    f'src/pipeline/data/.gitkeep',
    f'src/pipeline/features/.gitkeep',
    f'src/pipeline/model/.gitkeep',
    'main.py',
    'dvc.yaml',
    'app.py',
    'README.md',
    '.gitignore',
    'template.py',
    'requirements.txt',
    "Dockerfile",
    '.dockerignore'
]

for file in files:
    file = Path(file)
    print(file)
    dirs,_=os.path.split(file)
    if dirs != '':
        
        os.makedirs(dirs,exist_ok=True)
    if not os.path.exists(file) or os.path.getsize(file)==0 :
        with open(file,'w') as f :
            pass