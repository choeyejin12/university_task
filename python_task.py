
# numpy 는 파이썬의 리스트형 데이터를 연산처리르 위한 
#행렬로 변환하는 파이썬의 라이브러리 이다

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier , export_graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import graphviz

 

iris_data = load_iris()

X_train,X_test, Y_train,Y_test = train_test_split(
    iris_data.data,iris_data.target,random_state=11)


dTree = DecisionTreeClassifier(random_state=0)

dTree.fit(X_train,Y_train)

 

print("Train set score1 : {:.2f}".format(dTree.score(X_train, Y_train)))
print("Train set score1 : {:.2f}".format(dTree.score(X_test, Y_test)))
 

dTreeLimit = DecisionTreeClassifier(max_depth=3,random_state=0)
 

dTreeLimit.fit(X_train,Y_train)
 

print("Train set score2 : {:.2f}".format(dTreeLimit.score(X_train, Y_train)))

print("Train set score2 : {:.2f}".format(dTreeLimit.score(X_test, Y_test)))
 

export_graphviz(dTree, out_file="tree1.dot", class_names = iris_data.target_names,
                feature_names = iris_data.feature_names , impurity=True, filled=(True))
 

print("===============max_depth 의 제약이 없는 경우의 DEcision Tree 시각화=Random Forest =================")

 

with open("tree1.dot") as f:

    dot_graph = f.read()

    
graph = graphviz.Source(dot_graph)

graph.format='png'

graph.render("iris_tree",cleanup=True)





'''
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = load_iris()

iris_x = iris.data
iris_y = iris.target

x_train, X_test ,Y_train , Y_test = train_test_split(iris_x,iris_y, test_size=0.25,random_state=321)

print("x_train_shape ",x_train.shape)
print("x+test_shape",X_test.shape)

log_model = LogisticRegression(solver = 'liblinear',max_iter=1000)
log_model.fit(x_train, Y_train)

print("/n coef = ",log_model.coef_)
print( "/n intecept =",log_model.intercept_)

y_pred = log_model.predict(X_test)

acc = accuracy_score(Y_test,y_pred)
print('/n accuracy = ',acc)

print("2270003 yejin")

'''

'''
from email import header
from tkinter.tix import InputOnly
from importlib_metadata import _top_level_declared
import pandas as pd
from turtle import color
import matplotlib.pyplot as plt
from sys import flags
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

prestige = pd.read_csv ('prestige.csv', nrows=None)
print (prestige.head())

x = prestige[['education','women','prestige']]
y = prestige[['income']]
print("x dmen=",x.ndim,"y dmen=",y.ndim)
lr = LinearRegression()

lr.fit(x,y)
print('w = ',lr.coef_,'b=',lr.intercept_)
myincome = [[14.3,0.9,60.0]]
print('myincome predtiction =', lr.predict(myincome))
plt.figure(figsize=(12,4))
plt.subplots_adjust(wspace=0.5)
plt.subplot(131)
plt.title("2270003_yejin")
plt.xlabel("education")
plt.ylabel("income")
plt.scatter(prestige[['education']],prestige[['income']])
plt.subplot(132)
plt.title("2270003_yejin")
plt.xlabel("women")
plt.ylabel("income")
plt.scatter(prestige[['women']],prestige[['income']])
plt.subplot(133)
plt.title("2270003_yejin")
plt.xlabel("Prestige")
plt.ylabel("income")
plt.scatter(prestige[['education']],prestige[['income']])
plt.show()
'''
'''

x = pd.read_csv('score.csv',header=None).values
arr_x=x[1:,1:]
arr_x1=arr_x.astype('float')
subject_mean = arr_x1.mean(axis=0)
subject_mean_str=subject_mean.astype("str")

header = x[0,1:]+"subj. mean"

subject_mean_complete = np.vstack((header,subject_mean_str))
print(subject_mean_complete)

X = subject_mean_complete[0,:]
Y = subject_mean_complete[1,:]
plt.title("Subject Mean_2270003_choe ye jin")
bar=plt.bar(X,Y.astype("float"),color ='blue')
plt.ylim(0,100)
for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2.0, height,'%.1f' % height, ha = 'center', va = 'bottom',size=12)
plt.show()

student_mean = arr_x1.mean(axis=1)
student_mean_str = student_mean.astype("str")
header = x[1:,0]+"stu. mean"
student_mean_complete = np.vstack((header,student_mean_str))
print(student_mean_complete)
X = student_mean_complete[0,:]
Y = student_mean_complete[1,:]
plt.title("Subject Mean_2270003_choe ye jin")
bar = plt.bar(X,Y.astype('float'), color = 'red')
plt.ylim(0,100)
for rect in bar :
    height = rect.get_height()
    plt.text(rect.get_x()+ rect.get_width()/2.0, height,'%.1f'% height,ha = 'center', va = 'bottom',size=12)
plt.show()



digit = datasets.load_digits()
d=datasets.load_iris()
print(d.DESCR)
for i in range(0,len(d.data)):
    print(i+1,d.data[i],d.target[i])
plt.figure(plt.figimage)

a = int(input())
b = int(input())

print(a+b)
'''

'''

wh
A = np.array([ [10,20,30,40],[50,60,70,80]])
it = np.nditer(A,flags= ['multi_index'],op_flags=['readwrite'])ile not it.finished:
    idx = it.multi_index
    print('index =>', idx,' , value =>' , A[idx])

    it.iternext()

A = np.array ([2 , 6, 3, 1])
B = np.array([[2,6,3,1],[0,7,1,1]])

print("np.max(A) == ",np.max(A) )
print("np.argmax(A) == ",np.argmax(A) )

print("np.max(B) == ",np.max(B) )
print("np.argmax(B) == ",np.argmax(B) )
print(2270003,'choe ye jin')
'''
'''


import turtle # 외장함수 터틀 호출
import random 
##중심점을 기준으로 임의의 위치에 선을 긋고 아이콘을 달기
# 각 선의 끝 아이콘은 7가지 아이콘 중 랜덤사용
myTurtle, tX , tY,tColor,tSize, tShape = [None] *6 #비어있는 리스트 6개 생성

shapeList =[] #셰이프 리스트 생성
playerTurtles = []#거북 2차원 리스트
swidth,sheight = 500, 500

if __name__ == "__main__":
    turtle.title('거북 리스트 활용')
    turtle.setup(width =swidth + 50,height = sheight + 50) # 패딩 설정
    turtle.screensize(swidth,sheight)#.띄울 스크린 사이즈 설정

    shapeList = turtle.getshapes()
    for i in range(0,100):
        random.shuffle(shapeList)#셔플
        myTurtle = turtle.Turtle(shapeList[0])
        tX = random.randrange(-swidth / 2,swidth / 2)
        tY = random.randrange(-swidth / 2,swidth / 2)
        r = random.random(); g = random.random(); b = random.random()
        tSize = random.randrange(1, 3)
        playerTurtles.append([myTurtle,tX,tY,tSize,r,g,b])

    for tList in playerTurtles :
        myTurtle = tList[0]
        myTurtle.color((tList[4],tList[5],tList[6]))
        myTurtle.pencolor((tList[4],tList[5],tList[6]))
        myTurtle.turtlesize(tList[3])
        myTurtle.goto(tList[1],tList[2])
    turtle.done()
'''



