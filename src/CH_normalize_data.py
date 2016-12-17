import json

path = '../../Data/'

def normalize(filename):
    with open(path + filename + '.json', 'r') as f:
        s = f.read()
    s = s.lower()
    with open(path + filename + '.json', 'w') as f:
        f.write(s)

if __name__ == '__main__':
	normalize('merged_val')
	normalize('merged_train')