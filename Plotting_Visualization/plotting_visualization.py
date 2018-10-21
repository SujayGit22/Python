
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.plot([1,2,3,4])
plt.ylabel("some points")
plt.show()

plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])  # X and Y - axis(xmin,xmax,ymin,ymax)
#all sequences are converted to numpy internally
plt.show()

#example
t = np.arange(0.,5,0.2)

#red cross, blue square. green triangle
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

# Controlling line properties
plt.plot([1,2],linewidth=5.0)
line,  = plt.plot([1,2],'-')
line.set_antialiased(False)
plt.show()

lines = plt.plot([2,3,4,5])
plt.setp(lines,color='r',linewidth=2.0)
plt.show()

# working with multiple axis
def f(t):
    return np.exp(-t) * np.cos(2 * np.pi * t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(212)
plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
plt.show()

# Working with Text
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

''' Changing matplotlibrc settings
matplotlib uses matplotlibrc configuration files to customize all kinds of properties, which we call rc
settings or rc parameters. You can control the defaults of almost every property in matplotlib: figure
size and dpi, line width, color and style, axes, axis and grid properties, text and font properties and so on.
# Rc parameters can be changed directly
You can also dynamically change the default rc settings in a python script or interactively from the python
shell. All of the rc settings are stored in a dictionary-like variable called matplotlib.rcParams, which is 
global to the matplotlib package.
'''
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.color'] = 'r'
#or
mpl.rc('lines', linewidth=2,color='r')
#To restore defaults matplotlibrc values
mpl.rcdefaults()

''' Controlling interactive updating
The interactive property of the pyplot interface controls whether a figure canvas is drawn on every pyplot
command. If interactive is False, then the figure state is updated on every plot command, but will only be
drawn on explicit calls to draw(). When interactive is True, then every pyplot command triggers a draw.

isinteractive() # returns the interactive setting True | False
ion()           # turns interactive mode on
ioff()          # turns interactive mode off
draw()          # forces a figure redraw

When working with a big figure in which drawing is expensive, you may want to turn matplotlib’s interactive
setting o temporarily to avoid the performance hit: 
'''
# Example showing all the properties
fig = plt.figure()
fig.suptitle('Bold Figure',fontsize=14,fontweight='bold')

ax= fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes_title')
ax.set_xlabel('X-label')
ax.set_ylabel('Y-label')
ax.text(3, 8, 'boxed italics text in data coords', style='italic',
bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
#ax.text(3, 2, unicode('unicode: Institut f\374r Festk\366rperphysik', 'latin-1'))
ax.text(0.95, 0.01, 'colored text in axes coords',
verticalalignment='bottom', horizontalalignment='right',
transform=ax.transAxes,
color='green', fontsize=15)
ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
arrowprops=dict(facecolor='black', shrink=0.05))
ax.axis([0, 10, 0, 10])
plt.show()

# Building and defining properties and layout
import matplotlib.patches as patches

# Rectangle
left,width = .25,.5
bottom, height = .25,.5
right = left+width
top = bottom +height

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
# axes coordinates are 0,0 is bottom left and 1,1 is upper right
p = patches.Rectangle( (left, bottom), width, height,fill=False, transform=ax.transAxes, clip_on=False)
ax.add_patch(p)

ax.text(left, bottom, 'left top',horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
ax.text(left, bottom, 'left bottom',horizontalalignment='left',verticalalignment='bottom',transform=ax.transAxes)
ax.text(right, top, 'right bottom',horizontalalignment='right',verticalalignment='bottom',transform=ax.transAxes)
ax.text(right, top, 'right top',horizontalalignment='right',verticalalignment='top',transform=ax.transAxes)
ax.text(right, bottom, 'center top',horizontalalignment='center',verticalalignment='top',transform=ax.transAxes)
ax.text(left, 0.5*(bottom+top), 'right center',horizontalalignment='right',verticalalignment='center',rotation='vertical',transform=ax.transAxes)
ax.text(left, 0.5*(bottom+top), 'left center',horizontalalignment='left',verticalalignment='center',rotation='vertical',transform=ax.transAxes)
ax.text(0.5*(left+right), 0.5*(bottom+top), 'middle',horizontalalignment='center',verticalalignment='center',fontsize=20, color='red',transform=ax.transAxes)
ax.text(right, 0.5*(bottom+top), 'centered',
horizontalalignment='center',verticalalignment='center',rotation='vertical',transform=ax.transAxes)
ax.text(left, top, 'rotated\nwith newlines',horizontalalignment='center',verticalalignment='center',rotation=45,transform=ax.transAxes)
ax.set_axis_off()
plt.show()

