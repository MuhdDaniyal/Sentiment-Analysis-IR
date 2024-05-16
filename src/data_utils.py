import os

# Example of populating train_texts with training text data
train_texts = []

# Assuming you have text files in a directory named 'train'
train_dir = 'train'
for category in ['neg', 'pos']:
    category_dir = os.path.join(train_dir, category)
    for filename in os.listdir(category_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(category_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                train_texts.append(file.read())
