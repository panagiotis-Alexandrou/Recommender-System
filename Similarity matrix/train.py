from __future__ import absolute_import, division, print_function, unicode_literals

import sys, codecs, json, math, time, warnings, logging, os, shutil, subprocess, sqlite3, traceback, random
import numpy as np
import threading
from multiprocessing import Pool

LOG_FORMAT = ('%(levelname) -s %(asctime)s %(message)s')
logger = logging.getLogger( __name__ )
logging.basicConfig( level=logging.INFO, format=LOG_FORMAT )
logger.info('logging started')
exitFlag = 0
item_sim_matrix = np.full((2,2),-1,dtype=np.float16)
user_sim_matrix = np.full((2,2),-1,dtype=np.float16)
num_users = 0
num_items = 0
avg_user_rating = list()
avg_item_rating = list()
differences = list()
seqToSQL = dict()
sqlToSeq = dict()
progress = int(0)
progress2 = int(0)
itemUsers = dict()
conn = sqlite3.connect("comp3208_example.db")
c = conn.cursor()

# class myThread (threading.Thread):
#    def __init__(self, i, j):
#       threading.Thread.__init__(self)
#       self.i = i
#       self.j = j
#    def run(self):
      
#       sim(self.i,self.j)
      

# def sim(i, j):
     
#      if(i!=j):
#                     item_sim_matrix[i][j] = similI(i,j)
#                     item_sim_matrix[j][i] = item_sim_matrix[i][j]
                    
                    
                    
#      else:
#                       item_sim_matrix[i][j]=1
                      


def itemSimilarityMatrix():
        global item_sim_matrix 
        item_sim_matrix = np.full( (num_items, num_items), fill_value=-1.0, dtype=np.float16 )  
        allP = (pow(num_items,2) + num_items)/2
        for i in range(num_items):
            
            for j in range(i):
               global progress
               # progress+=1
               print(str((progress/((num_items*num_items +num_items)/2))*100)+" %")
               # thread = myThread(i,j)
               # thread.start()
               
               
               if(i!=j):
                    item_sim_matrix[i][j] = similI(i,j)
                    item_sim_matrix[j][i] = item_sim_matrix[i][j]
                    
                    progress+=1
                    
               else:
                      item_sim_matrix[i][j]=1
                      progress+=1
                


def similI (x,y):
        
        numerator = 0
        den1 = 0
        den2 = 0
        items = (seqToSQL[x],seqToSQL[y])
        elig = list()
        for user in itemUsers[seqToSQL[x]]:
              if (itemUsers[seqToSQL[y]].count(user)>0):
                    elig.append(user)
        #min_overlap = 100
     #    if len(elig) > min_overlap :
     #      return 0
        for i in elig:
            num = int(i)
            
                     
            numerator += (differences[num-1][items[0]])*(differences[num-1][items[1]])
            #maybe some mistakes here?
            den1 += pow((differences[num-1][items[0]]),2)
            den2 += pow((differences[num-1][items[1]]),2) 
        answer = 0
     
        if len(elig) !=0 and den1!=0 and den2!=0 : # right now incomputable similarities are labeled with 1
             answer = numerator/(pow(den1,0.5)*pow(den2,0.5))
        return answer
def buildModel():
     print("building model")
     
     c.execute("SELECT COUNT(DISTINCT ItemId) FROM example_table")
     row = c.fetchone()
     global num_items
     num_items = int(row[0])
     print("Num of items = "+ str(num_items))     
     c.execute("SELECT MAX(UserId) FROM example_table")
     row = c.fetchone()
     global num_users
     num_users = int(row[0])
    #  print("Num of users = "+ str(num_users))
     c.execute("SELECT AVG(Rating) FROM example_table GROUP BY UserId ")
     global avg_user_rating
     avg_user_rating = c.fetchall()
     c.execute("SELECT AVG(Rating) FROM example_table GROUP BY ItemId ")
     global avg_item_rating
     avg_item_rating = c.fetchall()
     j = 0
     global differences
     for i in range(num_users):
           differences.append(dict())
           c.execute("SELECT ItemId,Rating FROM example_table WHERE UserId={}".format(i+1))
           rows = c.fetchall()
           for pair in rows:
                 differences[i][pair[0]] = pair[1]-avg_user_rating[i][0]
     c.execute("SELECT MAX(ItemId) FROM example_table ")
     maxItem = c.fetchone()
     for i in range(maxItem[0]+1): #max item
            
            c.execute("SELECT ItemId,UserId,Rating FROM example_table WHERE ItemId={}".format(i))
            entry = c.fetchall()
            if(entry != []):
                  
                  users= list()
                  for tuplee in entry:
                        users.append(tuplee[1])
                  itemUsers[entry[0][0]] = users
                  seqToSQL[j] = entry[0][0]
                  sqlToSeq[entry[0][0]] = j
                  j = j+1
     print("model built!")
    

def similarityUser ():
        global user_sim_matrix 
        user_sim_matrix = np.full( (num_users, num_users), fill_value=-1.0, dtype=np.float16 )  
        allP = (pow(num_users,2) + num_users)/2
        for i in range(num_users):
            
            for j in range(i):
               global progress2
               # progress+=1
               print(str((progress2/((num_users*num_users +num_users)/2))*100)+" %")
               # thread = myThread(i,j)
               # thread.start()
               
               
               if(i!=j):
                    user_sim_matrix[i][j] = simu(i,j)
                    user_sim_matrix[j][i] = user_sim_matrix[i][j]
                    
                    progress2+=1
                    
               else:
                      user_sim_matrix[i][j]=1
                      progress2+=1
                
def simu (x,y):
        
        numerator = 0
        den1 = 0
        den2 = 0
        items = list()
        for item in itemUsers.keys():
              if(itemUsers[item].count(x+1)>0 and itemUsers[item].count(y+1)>0):
                    items.append(item)
        for i in items:
                numerator += (differences[x][item])*(differences[y][item])
                den1 += pow((differences[x][item]),2)
                den2 += pow((differences[y][item]),2)
        answer = numerator/(pow(den1,0.5)*pow(den2,0.5))
        return answer


def getNeighboursNew(item,user):
     neighbours = list()
     k = 5
     for i in range(len(item_sim_matrix[item])):
               a = (i,item_sim_matrix[item][i])
               if(itemUsers[seqToSQL[i]].count(user)>0 and a[1]>0):
                    neighbours.append(a)
     neighbours.sort(key=lambda a : a[1],reverse=True)
     #print(neighbours)
     while(neighbours[k-1][1]>0.5):
          k+=1
     return(neighbours[0:k])
def predictNew(item,user):
      answer = avg_item_rating[sqlToSeq[item]]
      num = 0
      den = 0
      itemNeighbours = getNeighbours(item=item,user=user)
      userNeighbours = getNeighboursNew(item=item,user=user)
      avgI =0
      try:
          avgI  = avg_item_rating[sqlToSeq[item]]
      except Exception as e:
            print(e)
      
      for i in itemNeighbours:
            for u in userNeighbours:
                  num+=user_sim_matrix[user-1][u]*item_sim_matrix[sqlToSeq[item]]*(differences[u-1][item])
                  den+=user_sim_matrix[user-1][u]*item_sim_matrix[sqlToSeq[item]]
      if(num!=0 and den!=0):
            answer = num/den
      return round(answer)






def fillSubmission():
     writer = open("results.csv","a")
     readHandle = codecs.open( 'test_100k_withoutratings_new.csv', 'r', 'utf-8', errors = 'replace' )
     listLines = readHandle.readlines()
     readHandle.close()
     for strLine in listLines:
          if len(strLine.strip()) > 0 :
               listParts = strLine.strip().split(",")
               if len(listParts) == 3:
                     user = int(listParts[0])
                     item = int(listParts[1])
                     prediction = 0
                     try:
                         seq = sqlToSeq[item]
                         prediction = round(predict(user,seq))
                     except Exception as e:
                         print(e)
                         prediction = round(avg_user_rating[user-1][0])
                     timestamp = listParts[2]
                     
                     print("{},{},{},{}\n".format(user,item,prediction,timestamp))
                     writer.write("{},{},{},{}\n".format(user,item,prediction,timestamp))
               else :raise Exception( 'failed to parse csv : ' + repr(listParts))
def getNeighbours(item,user):
     neighbours = list()
     k = 5
     for i in range(len(item_sim_matrix[item])):
               a = (i,item_sim_matrix[item][i])
               if(itemUsers[seqToSQL[i]].count(user)>0 and a[1]>0):
                    neighbours.append(a)
     neighbours.sort(key=lambda a : a[1],reverse=True)
     if (len(neighbours)>k):
          while(neighbours[k-1][1]>0.4 and len(neighbours)>k):
               k+=1
     return(neighbours[0:k])




def predict(user,itemseq):
     

     neighbours = getNeighbours(itemseq,user=user)
     numerator = 0
     denominator = 0
     rating = 0
     
     for pair in neighbours:
          print(pair)
          print(user)
          print(seqToSQL[pair[0]])
          row = None
          try:
                row = differences[user-1][seqToSQL[pair[0]]]+avg_user_rating[user-1][0]
          except Exception as e:
                print(e)
                row = None
          if(type(row) != type(None)):
               rating = round(row,4)
               sim = round(pair[1],4)    
               print(rating)
               numerator += sim*rating
               denominator += sim
     answer = 0
     try:
          answer = round(numerator,4)/round(denominator,4)
     except Exception as e:
          print(e)
          answer = round(avg_user_rating[user-1][0])
     return answer


          
if __name__ == '__main__':

    logger.info( 'loading training set and creating sqlite3 database' )
    buildModel()
    
    itemSimilarityMatrix()
    #similarityUser()
    fillSubmission()
#     item = 1
#     user = 1
#     while True:
#         t = False
#         while(not t):
#                try:
#                  print("write a user for a prediction")
#                  user = int(input())
#                  print("write an item for a prediction")
#                  item = int(input())
#                  t = True
#                except Exception as e:
#                      print(e)
#         while (user < 0 or user>num_users or not(item in sqlToSeq)): 
#              print("your inputs were not in range of the sets \n do you want to terminate the session? y -> yes/n -> no")
#              prompt = input()
#              if (prompt == "y"):
#                   break
#              else:
#                     t = False
#                     while(not t):
#                            try:
#                              print("write a user for a prediction")
#                              user = int(input())
#                              print("write an item for a prediction")
#                              item = int(input())
#                              t = True
#                            except Exception as e:
#                                  print(e)
#         search = sqlToSeq[item]
#         print(predict(user,search,item))