import pickle as p

with open('train_data.data', 'rb') as f:
    train_data = p.load(f)

pA = 0  
pNotA = 0  
SPAM = 1  
NOT_SPAM = 0  
trainPositive, trainNegative = {}, {}  
positive_total, negative_total = 0, 0


def train():  
    global pA, pNotA  
    total = 0.  
    num_spam = 0.  
    for (email, label) in train_data:  
        calculate_word_frequencies(email, label)  
        if label == SPAM:  
            num_spam += 1  
        total += 1  
    pA = num_spam / total  
    pNotA = 1 - pA  

def calculate_word_frequencies(body, label):  
    global trainPositive, trainNegative, positive_total, negative_total  
    for word in body:  
        if label == SPAM:  
            trainPositive[word] = trainPositive.get(word, 0) + 1  
            positive_total += 1  
        else:  
            trainNegative[word] = trainNegative.get(word, 0) + 1  
            negative_total += 1  

def calculate_P_Bi_A(word, label):  
    # P(Bi|A)  
    if label == SPAM:  
        return (trainPositive.get(word, 0) + 1) / positive_total  
    else:  
        return (trainNegative.get(word, 0) + 1) / negative_total

def calculate_P_B_A(text, label):  
    # P(B|A)  
    result = 1.0  
    for word in text.lower().split():  
        result *= calculate_P_Bi_A(word, label)  
    return result  

def classify(email):  
    isSpam = pA * calculate_P_B_A(email, SPAM)   
    notSpam = pNotA * calculate_P_B_A(email, NOT_SPAM)  
    return isSpam > notSpam
