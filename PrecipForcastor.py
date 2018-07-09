import time
from sklearn.externals import joblib
from scipy import interpolate
import numpy as np
import pandas as pd
import datetime
import math
from multiprocessing.dummy import Pool as ThreadPool
rootdir='C:\\Users\\xia\\logregres\\'
modelSaveModelPath='C:\\Users\\xia\\logregres\\modelsave\\'
ecdir='K:\\'
product_dir='M:\\'

##时间转换函数time ticks##
def timetoticks(t,geshi):
    ticks=time.mktime(time.strptime(t,geshi))
    return ticks

def tickstotime(ticks,geshi):
    timetime=time.strftime(geshi,time.localtime(ticks))
    return timetime
##时间转换函数time ticks##

######时间加减函数#######
def switchtime(t,geshi0,dt,geshi1): #t为时间 geshi为时间格式 dt为时间的变化量（天）
    newtime=tickstotime((timetoticks(t,geshi0)+3600*24*dt),geshi1)
    return newtime

def linear(x1,x2,y1,y2,x):###一元一次线性外推
    y=y2-((x2-x)*(y2-y1)/(x2-x1))
    return y

def searchindex(text,start,search):
    position=[]
    while True:
        try:
            ii=text.index(search,start)
        except ValueError:
            break
        else:
            position.append(ii)
            start=ii+1
    return position

#############zhuguan precipitation Clustering################        
def raincluster():
    fmap=pd.read_excel(rootdir+'JxNationalStation.xlsx',skiprows=1)
    t1='赣中：17'
    c1='莲花 永新 夏坪 井冈山 遂川 万安 兴国 宁都 泰和 吉安县 吉水 永丰 乐安 南城 南丰 黎川 广昌'
    t2='浙赣铁路东段：17'
    c2='万年 进贤 丰城 金溪 东乡 余江 鹰潭 贵溪 弋阳 横峰 铅山 上饶 上饶县 广丰 玉山 资溪'
    t3='赣南：16'
    c3='石城 于都 瑞金 赣县 上犹 崇义 南康 大余 信丰 会昌 安远 龙南 全南 定南 寻乌'
    t4='赣北中南部：16'
    c4='铜鼓 修水 靖安 安义 永修 奉新 新建 宜丰 高安 南昌 南昌县 余干 波阳  乐平 德兴 婺源'
    t5='浙赣铁路西段：14'
    c5='萍乡 宜春 分宜 安福 万载 上高 新余 峡江 新干 崇仁 樟树 宜黄'
    t6='赣北沿江：12'
    c6='武宁 都昌 星子 德安 庐山 九江 瑞昌 湖口 彭泽 景德镇'
    t=[t1,t2,t3,t4,t5,t6]
    c=[c1,c2,c3,c4,c5,c6]
    cluster=[]
    for i in range(len(t)):
        clt=[]
        for cc in c[i].split():
            try:
                df=fmap[fmap['站名']==cc]
                print(df.iloc[0,2])
            except:
                print(cc)
            else:
                clt.append(df.iloc[0,2])
        cluster.append(clt)
    return cluster

def linint2(lon,lat,var,stlon,stlat):
    LonRange=[];LatRange=[];newvar=[]
    for m in stlon:
        lon.append(m);lon.sort()
        i=lon.index(m);a=lon[i-1];b=lon[i+1]
        LonRange.append([a,b])
        del lon[i]
    for m in stlat:
        lat.append(m);lat.sort()
        i=lat.index(m);a=lat[i-1];b=lat[i+1]
        LatRange.append([a,b])
        del lat[i]
    for i in range(len(stlon)):
        x0=stlon.iloc[i];y0=stlat.iloc[i]
        x1,x2=LonRange[i];y1,y2=LatRange[i]
        p11=var.loc[y1,x1];p12=var.loc[y2,x1]
        p21=var.loc[y1,x2];p22=var.loc[y2,x2]
        #p0=np.matrix([[1-x0,x0]])*np.matrix([[p11,p21],[p12,p22]])*np.matrix([[1-y0],[y0]])
        p01=(p21-p11)/(x2-x1)*(x0-x1)+p11
        p02=(p22-p12)/(x2-x1)*(x0-x1)+p12
        p0=(p02-p01)/(y2-y1)*(y0-y1)+p01
        newvar.append(p0)
    return newvar


#########main###########
#######part1 interpolate########
def main(forcastor):
    nowtime=datetime.datetime.now().strftime('%y%m%d%H')
    pasttime=(datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%y%m%d%H')
######date为模式起报时间；forcast为预报时效#####
    forcast=['012','015','018','021','024']
    if float(nowtime[-2:])<15:
        date=pasttime[:-2]+'20'
    else:
        date=nowtime[:-2]+'08'
    varlist=['Q','R','Td','W','TP'];levlist=['925','850','700','500']
    lat=np.linspace(45,15,121);lon=np.linspace(90,125,141)
    fmap=pd.read_table(rootdir+'jx86.txt',header=None,sep='\s+',index_col=0)
    fmap.columns=['lon','lat']
    df=pd.DataFrame(index=fmap.index)
    for var in varlist:   
        for lev in levlist:
            print(var+lev)
            if var=='Td':
                path='K:\\physic\\'+var+'\\'+lev+'\\'+date+'.'+forcastor
            elif var=='TP':
                path='K:\\'+var+'\\r12\\'+date+'.024'
                lat=np.linspace(45,15,241);lon=np.linspace(90,125,281)
            else:
                path='K:\\'+var+'\\'+lev+'\\'+date+'.'+forcastor
            if var=='Td':
                ftd=open(path,'r');ftd=ftd.read()
                ftd=ftd.split()[22:]
                ftd=np.array(ftd).astype(float).reshape(121,141)
                locals()[var+lev]=pd.DataFrame(ftd,index=lat,columns=lon)
            else:
                locals()[var+lev]= pd.read_table(path,header=None,sep='\s+',skiprows=5)
                locals()[var+lev].index=lat
                locals()[var+lev].columns=lon
            locals()[var+lev]=locals()[var+lev][lat[-1]:lat[0]:-1]
        #print(locals()[var+lev].loc[27.25,115:118])
            newvar=linint2(list(lon),list(lat),locals()[var+lev],fmap['lon'],fmap['lat'])
            df0=pd.DataFrame(newvar,index=fmap.index,columns=[var+lev])
            df[var+lev]=df0[var+lev]
    df['TP']=df['TP'+levlist[0]]
    for lev in levlist:
        del df['TP'+lev]
    del df.index.name
    df.to_csv(rootdir+forcastor+'.csv')
    return df  
    
##实际预报处理部分##    
def predict(listyp,listp0):
    ff=open(rootdir+'ypyp.txt','w')
    #a=listyp[0:3];a.append(listyp[6]);b=listp0[0:3];b.append(listp0[6])
    #a=listyp[3:];b=listp0[3:]
    a=listyp;b=listp0
    yp=np.array(a);p0=np.array(b).astype(float)
    rt=[]
    for i in range(np.shape(yp)[1]):     #时间循环
        flag=[]
        for ilabel in range(np.shape(yp)[0]):   #label循环
            if yp[ilabel][i]>p0[ilabel]:
                flag.append(1)
            else:
                flag.append(0)
            ff.write(str(flag[ilabel])+' ')
        for ilabel in range(np.shape(yp)[0]):
            ff.write(str(yp[ilabel][i])+' ')
        ff.write('\n') 
        ####-1无雨 0小雨 1中雨 2大雨 3暴雨
        if flag.count(1)>=1:
            j=searchindex(flag,0,1)  
            result=max(j)    
        else:
            result=-1
        rt.append(result)    
    return rt

########PrecipForcastor.exe#########
#####Part1 prepare physics##########
nowtime=datetime.datetime.now().strftime('%y%m%d%H')
pasttime=(datetime.datetime.now()-datetime.timedelta(days=1)).strftime('%y%m%d%H')
if float(nowtime[-2:])<15:
    date=pasttime[:-2]+'20'
else:
    date=nowtime[:-2]+'08'
forcast=['012','015','018','021','024']
fmap=pd.read_table(rootdir+'jx86.txt',header=None,sep='\s+',index_col=0)
fmap.columns=['lon','lat']
tpool=ThreadPool(8)
dflist=tpool.map(main,forcast)
tpool.close()
tpool.join()
df=sum(dflist)/len(forcast)
print(df)
#####Part2 forcast#######
#######select model######
cluster=raincluster()
fout=open(rootdir+'fout_now.txt','w')
for c in cluster:
    yp=[]
    for ilabel in range(4):
        ml= joblib.load(modelSaveModelPath+str(c[0])+'_cluster_'+str(ilabel)+'Pca_Logistic.model')
        if ilabel==3:
            yp.append(ml.predict_proba(df.loc[c,df.columns[:-1]])[:,1])
        else:
            yp.append(ml.predict_proba(df.loc[c])[:,1])
    p0=open(modelSaveModelPath+str(c[0])+'-cluster_p0.txt','r')
    p0=p0.readlines()
    rf=predict(yp,p0)
    ##########概率预报转化成数值预报###########
    arryp=np.array(yp).astype(float);arrp0=np.array(p0).astype(float)
    for i,st in enumerate(c):
        fout.write(str(st)+' ')
        fout.write(tickstotime((timetoticks(date,'%y%m%d%H')+3600*24),'%y%m%d%H'))
        if rf[i]==-1:
            r12=0
        elif rf[i]==0:
            pb0=arrp0[0];pb=arryp[0,i]
            r12=linear(pb0,1,0.01,10,pb)
        elif rf[i]==1:
            pb0=arrp0[1];pb=arryp[1,i]
            r12=linear(pb0,1,10,25,pb) 
        elif rf[i]==2:
            pb0=arrp0[2];pb=arryp[2,i]
            r12=linear(pb0,1,25,50,pb)
        elif rf[i]==3:
            pb0=arrp0[3];pb=arryp[3,i]
            r12=linear(pb0,1,50,140,pb)    
        fout.write(' '+str(r12)+'\n')
        print(str(st)+'_over')
fout.close()
#生成micaps类格式文件
df=pd.read_table(rootdir+'fout_now.txt',header=None,sep='\s+',index_col=0)
df.columns=['datetime','r12']
df.insert(0,'lon',fmap['lon'])
df.insert(1,'lat',fmap['lat'])
df.insert(2,'height',0)
del df.index.name
del df['datetime']
fmicaps=open(product_dir+date+'.024','w')
fmicaps.write('diamond 3 '+date+'_forcast-24h\n')
fmicaps.write('6 1 6 22 -1 0 1 50 0 1 93\n')
fmicaps.write(df.to_string(header=False))
fmicaps.close()