import pickle

def load_data(location):
    with open(location, 'rb') as file:
        data = pickle.load(file)
    return data

def save_data(data, location):
    with open(location, 'wb') as file:
        pickle.dump(data, file)