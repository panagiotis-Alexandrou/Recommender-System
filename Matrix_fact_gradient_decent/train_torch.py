import torch
import numpy as np
import codecs
import random

# uses associated with items,ratings and timestamps
records = dict()
#file which contains the ratings
training_file = 'train_100k_withratings_new.csv'
testing_file = 'test_100k_withoutratings_new.csv'
output_file = 'results.csv'
#items to users
i_u = dict()

# imports the 20m csv file
def importData():
    readHandle = codecs.open(training_file, 'r', 'utf-8', errors='replace')
    listLines = readHandle.readlines()
    readHandle.close()
    buildModel(data=listLines)

# receives data and creates the model
def buildModel(data):
    for datum in data:
        if len(datum.strip()) > 0:
            # userid, itemid, rating, timestamp
            datumSplit = datum.strip().split(',')
            if len(datumSplit) == 4:
                u_id, i_id, rating, ts = int(datumSplit[0]), int(datumSplit[1]), float(datumSplit[2]), int(datumSplit[3])
                li = list()
                li.append(i_id)
                li.append(rating)
                li.append(ts)
                if(not (u_id in records)):
                    records[u_id] = list()
                records[u_id].append(li)
                if(not (i_id in i_u)):
                    i_u[i_id] = list()
                i_u[i_id].append(u_id)
            else:
                # Exception if the line is not in the correct format
                raise Exception('failed to parse csv : ' + repr(datumSplit))

# retuns the rating of user item pair
# returns -1 if it does not exist
def rating(u, i):
    for li in records[u]:
        item = li[0]
        rate = li[1]
        if(item == i):
            return rate
    return -1

def train():

    factor_number = 40
    learning_rate = 0.001
    regularization = 0.001
    pairs_size = 1000
    iterations = 200

    counter = 1

    
    user_vec = torch.randn(size=(len(records), factor_number)) 
    item_vec = torch.randn(len(i_u), factor_number) 
   
    for i in range(iterations):
        print(counter)
        counter += 1
        pairs = set()
        # create random shuffled set of user - item pairs
        for j in range(pairs_size):
            # random item
            item = random.randint(1, len(i_u))
            while(not item in i_u):
                item = random.randint(1, len(i_u))
            # random user where Rating(user,item) is known

            subIndex = int((len(i_u[item])-1)*0.7)

            index = random.randint(0, subIndex)
            user = i_u[item][index]
            pairs.add(tuple((user, item)))

        #loop through all pairs
        for pair in pairs:
            user = pair[0] - 1
            item = pair[1] - 1
            predicted = torch.dot(item_vec[item], user_vec[user])
            actual = rating(user+1, item+1)
            if actual != -1:
                error = actual - predicted
                user_vec[user] += learning_rate * (error * item_vec[item] - regularization * user_vec[user])
                item_vec[item] += learning_rate * (error * user_vec[user] - regularization * item_vec[item])
    # checkMAE(uservex=user_vec,itemvex=item_vec)
    fillSubmission(user_vec,item_vec)
    
def predict():
    
    print("write your user for prediction")
    user = 1
    item = 1
    try:
        user = int(input())
    except Exception as e:
        print(e)
    try:
        item = int(input())
    except Exception as e:
        print(e)
    pred = tuple((user-1,item-1))
    return pred


def fillSubmission(uservex,itemvex):
     
     writer = open(output_file,"a")
     readHandle = codecs.open(testing_file , 'r', 'utf-8', errors = 'replace' )
     listLines = readHandle.readlines()
     readHandle.close()
     for strLine in listLines:
          if len(strLine.strip()) > 0 :
               listParts = strLine.strip().split(",")
               if len(listParts) == 3: # originally 3
                     user = int(listParts[0])
                     item = int(listParts[1])
                     prediction = 0
                     try:
                         tens = torch.dot(itemvex[item-1],uservex[user-1])
                         prediction = round(float(tens))
                         if prediction > 5:
                             prediction = 5
                     except Exception as e:
                         print(e)
                         prediction = 3.0
                     timestamp = listParts[2]
                     
                     print("{},{},{},{}\n".format(user,item,prediction,timestamp))
                    
                     writer.write("{},{},{},{}\n".format(user,item,prediction,timestamp))
               else :raise Exception( 'failed to parse csv : ' + repr(listParts))
     writer.close()

def checkMAE(uservex,itemvex):
    total = 0
    count = 0
    
    for item in i_u:
        subIndex = int((len(i_u[item])-1)*0.7)

        for user in range(subIndex,len(i_u[item])):
            rate = rating(user,item)
            prediction = torch.dot(uservex[user-1],itemvex[item-1])

            if(rate != -1):
                count+=1
                total+=abs(prediction-rate)

    return total/count
if __name__ == '__main__':
    importData()
    
    train()
    
    