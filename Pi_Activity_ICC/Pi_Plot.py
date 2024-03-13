import matplotlib.pyplot as plt
def piiteration(max):
    x=[]
    y=[]
    sum=0
    for i in range(1,max+1):
        sum=sum+((+1)**(i+1))*(1/(2*i+1))
        y.append(sum*4)
        x.append(i)
    plt.plot(x,y)
    plt.plot(y)
    plt.show()
    print(x)
piiteration(1000000)
