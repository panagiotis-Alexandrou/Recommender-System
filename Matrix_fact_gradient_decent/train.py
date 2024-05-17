import numpy as np
import codecs

#Code has been influnced by code from: https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea

# users pointing towards tuples of items,ratings and timestamps
records = dict()

#items to users
i_u = dict()

numberOfItems = 0
numberOfUsers = 0

#ratings matrix initialized
ratings = np.zeros((1,1))

# hold the indices of users and items that have a rating and will be included in the training set 
sample_users = np.zeros((1,1))
sample_items = np.zeros((1,1))
# number of samples gathered for traversal
samples_num = 0
#user item vectors
user_mat = np.zeros((1,1))
item_mat = np.zeros((1,1))

# user item biases 
# considering that users have a bias when rating most movies, some are generous and usually give a high rating while some others are harsh and rate most with low
# Simultaneously, in movies, this can be seen. For example, a movie can be rated highly by many users and some others have a bias to be rated lower by most users.
# global_bias is the average rating as a base of the prediction. While predictions are unique, the mean value is a good approach for a base value, considering standard deviation.
user_bias= np.zeros((1,1))
item_bias= np.zeros((1,1))
global_bias= 0

#regularisation variables
user_bias_reg = 0.0
item_bias_reg = 0.0
user_fact_reg = 0.0
item_fact_reg = 0.0

# learning rate at 0.001 after experimenting between  0.001 - 0.1
learning_rate = 0.001

# number of latent factors after experimenting between 20-80
factor_number = 40


#file which contains the ratings and test data
training_file = 'train_20m_withratings_new.csv'
testing_file = 'test_20m_withoutratings_new.csv'
counter = 0
output_file = 'results'


#imports the 20m csv file
def importData():
    global ratings,sample_items,sample_users,samples_num,numberOfUsers,numberOfItems
    readHandle = codecs.open( training_file, 'r', 'utf-8', errors = 'replace' )
    listLines = readHandle.readlines()
    readHandle.close()

    # matrix dimensions 
    shape = buildModel(data=listLines)
    print(shape)

    numberOfUsers = shape[0]
    numberOfItems = shape[1]

    print(numberOfUsers)
    print(numberOfItems)

    # creates the matrix of the correct shape
    ratings = np.zeros(shape)
    
    for user in records:
        for li in records[user]:

            item = li[0]
            
            ratings[user-1][item-1] = li[1]
    # rows and columns where ratings are not zero for trainning
    sample_users,sample_items = ratings.nonzero()
    
    #number of sample data
    samples_num = len(sample_users)
    


#receives data and creates the model
def buildModel(data):
    maxU = 0
    maxI = 0
    for datum in data:
        
        if len(datum.strip()) > 0 :
			# userid, itemid, rating, timestamp
            datumSplit = datum.strip().split(',')
            if len(datumSplit) == 4 :
                u_id,i_id,rating,ts = int(datumSplit[0]),int(datumSplit[1]),float(datumSplit[2]),int(datumSplit[3])
                li = list()
                li.append(i_id)
                li.append(rating)
                li.append(ts)

                if(u_id > maxU):
                    maxU = u_id
                if(i_id > maxI):
                    maxI = i_id
                    

                if(not (u_id in records)):
                    records[u_id] = list()
                records[u_id].append(li)
                if(not (i_id in i_u)):
                    i_u[i_id] = list()
                i_u[i_id].append(u_id)
            else :
                # Exception if the line is not in the correct format
                raise Exception( 'failed to parse csv : ' + repr(datumSplit))
            
    return tuple((maxU,maxI))

#retuns the rating of user item pair
#returns -1 if it does not exist
def rating(u,i):
    for li in records[u]:
        item = li[0]
        rate = li[1]
        if(item == i):
            return rate
    return -1

#train model with  stochastic gradient decent approach
def train(iter):
    global global_bias,item_bias,user_bias,user_mat,item_mat,learning_rate,factor_number,ratings


    #random user and item vector values from 0-1 (Nusers X NFactors) and (Nitems X NFactors) normalised by the number of latent factors to make the outcome close to the rating
    user_mat = np.random.normal(scale=1/factor_number,size=(numberOfUsers,factor_number))
    item_mat = np.random.normal(scale=1/factor_number,size=(numberOfItems,factor_number))

    user_bias = np.zeros(numberOfUsers)
    item_bias = np.zeros(numberOfItems)
    global_bias = np.mean(ratings[np.where(ratings != 0)])

    partial_training(iter)

# train on partial trainning data 
def partial_training(times):
    global user_bias,user_bias_reg,item_bias,item_bias_reg,user_bias_reg
    for i in range(times):
        print("iteration No "+str(i))
        # prepare trainning set of indices from number of indices
        training_idc = np.arange(samples_num)
        #randomly shuffle the trainning indices
        np.random.shuffle(training_idc)

        for idx in training_idc:
            user = sample_users[idx]
            item = sample_items[idx]

            #predict for one user,item pair
            prediction = predict(user,item)

            #error = actual - prediction
            error = rating(user+1,item+1) - prediction

            #update factors Pu = Pu + γ (error * Qi - λ * (Pu)) & Qi = Qi + γ (error * Pu - λ * (Qi))
            user_mat[user] += learning_rate * (error * item_mat[item] - user_fact_reg *user_mat[user])
            item_mat[item] += learning_rate * (error * user_mat[user] - item_fact_reg *item_mat[item])

            #update biases with the same formula as the user/item vectors
            user_bias[user] += learning_rate*  (error - user_bias_reg * user_bias[user])
            item_bias[item] += learning_rate*  (error - item_bias_reg * item_bias[item])

# predict a single user,item rating
def predict(user,item):
    global global_bias

    #   starts the prediction from a base of the mean value of non zero values in the matrix and bias values
    prediction = global_bias + user_bias[user] + item_bias[item]
    #   adds the dot product of user vector and transpose of item vector (Rui = Pu.QiT) normalized by the number latent factors.
    #   that allows this value to predict the deviation needed from the mean value, either negative or positive.
    prediction += user_mat[user].dot(item_mat[item].T)
    return prediction


# manual input testing of predictions
def receive():
    
    print("write your user for prediction")
    user = 1
    item = 1

    try:
        user = int(input())
    except Exception as e:
        print(e)

    print("write your item for prediction")

    try:
        item = int(input())
    except Exception as e:
        print(e)
    
    #offset
    pred = tuple((user-1,item-1))

    return pred


# retrieve all test data and fill the outputfile with comma separated tuples of (user_id,item_id,rating,timestamp)
def fillSubmission():
     global counter

     counter+=1
     if(counter != 0):
         out_file = str(output_file) +str(counter)+str(".csv")
     #create the outputstream
     writer = open(out_file,"a")

     # read all lines from the testing file, then close
     readHandle = codecs.open(testing_file , 'r', 'utf-8', errors = 'replace' )
     listLines = readHandle.readlines()
     readHandle.close()

     for strLine in listLines:
          
          if len(strLine.strip()) > 0 :
               
               listParts = strLine.strip().split(",")
               if len(listParts) == 3:
                     
                     #offset attributes for 0-indexed matrices
                     user = int(listParts[0])-1
                     item = int(listParts[1])-1
                     prediction = 0
                     try:
                         
                         pred = predict(user,item)
                         print(pred)
                         # round prediction to the nearest int
                         prediction = round(float(pred))

                         if prediction > 5:
                             prediction = 5

                     except Exception as e:
                         # if a problem occurs in the prediction, give the median of 0-5
                         print(e)
                         prediction = 3.0
                         
                     timestamp = listParts[2]

                     writer.write("{},{},{},{}\n".format(user+1,item+1,prediction,timestamp))
               else :raise Exception( 'failed to parse csv : ' + repr(listParts))
     writer.close()


if __name__ == '__main__':
    
    importData()
    print(ratings)
    
    train(1)
    partial_training(24)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()
    partial_training(5)
    fillSubmission()

    


    
    
        