import chardet


def detect_encoding(filename):
    with open(r'D:\Users\HP\Desktop\ade\20\data20video.csv' , 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']


encoding = detect_encoding(r'D:\Users\HP\Desktop\ade\20\data20video.csv' )
print(f'Detected encoding: {encoding}')
