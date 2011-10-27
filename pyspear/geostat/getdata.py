from numpy import *
 
# Download datafile
#import urllib
#urllib.urlretrieve('http://www.ai-geostats.org/fileadmin/Documents/Data/walker_01.dat',filename='walker_01.dat')

# Whether to thin dataset; definitely thin it if you're running this example on your laptop!
#thin = False
thin = True

l = file('walker_01.dat').read().splitlines()[8:-1]
a = array([fromstring(line,sep='\t') for line in l])
if thin:
    a=a[::5]
ident,x,y,v,u,t=a.T
mesh = vstack((x,y)).T
