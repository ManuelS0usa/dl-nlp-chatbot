from .utils import *
from config import *
from torch.utils.data import Dataset


class Training_Data(Dataset):
    """ Apply NLP data preparation techniques and build dataset for training """

    def __init__(self, data):
        self.data = data
        self.x_data = []
        self.y_data = []
        self.n_samples = None
        
    def __getitem__(self, index):
        """ Support indexing such that dataset[i] can be used to get i-th sample """
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """ We can call len(dataset) to return the size """
        return self.n_samples

    def get_x_data(self):
        return self.x_data
        
    def get_y_data(self):
        return self.y_data

    def get_input_size(self):
        return len(self.x_data[0])

    def transform(self):
        all_words = []
        tags = []
        xy = []
        # loop through each sentence in our intents patterns
        for intent in self.data['intents']:
            tag = intent['tag']
            tags.append(tag)
            for pattern in intent['patterns']:
                w = tokenize(pattern)  # tokenization
                all_words.extend(w)
                xy.append((w, tag))

        # stemming, lower case and ponctuation removal
        all_words = [stem(w) for w in all_words if w not in IGNORE_WORDS]  

        # remove duplicates and sort
        all_words = sorted(set(all_words))
        tags = sorted(set(tags))
        return all_words, tags, xy

    def build(self):
        """ Build target and input features """
        all_words, tags, xy = self.transform()
        X_train = []
        y_train = []
        for (pattern_sentence, tag) in xy:
            bag = bag_of_words(pattern_sentence, all_words)
            X_train.append(bag)

            label = tags.index(tag)  # position in tags list
            y_train.append(label)  # cross entropy loss(thats why we dont use one hot encoder)

        self.x_data = np.array(X_train)
        self.y_data = np.array(y_train)
        self.n_samples = len(self.x_data)
        
        return all_words, tags, xy