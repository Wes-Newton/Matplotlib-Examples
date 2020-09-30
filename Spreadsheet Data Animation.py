# -*- coding: utf-8 -*-


class Ani_Plot(object):
    
    """
    This is mostly if not entirely original code.
    
    This class creates a matplotlib animated chart from spreadsheet data.
    It takes the worksheet-ws, axis-left, axis-right, and the column headers
    to plot on left axis-Prim_cols, the column headers to plot on the 
    right axis-Sec_cols, and chart title.
    
    Please see the question below and vote to unhide.
    
    https://stackoverflow.com/questions/57660890/improving-performance-
    of-matplotlib-animation-plot   
    
 
    """
    
    def __init__(self, ax, ax2, ws, Prim_cols, Sec_cols, title):
        
        """Initialization routine for chart animation.  Sets up
        X and Y values that will be plotted."""        
        app.update_status("Retrieving data from worksheet")
        self.title = title        
        self.ax = ax
        self.ax2 = ax2
        self.Column_Header = []
        #last_col = ws.max_column
        self.last_row = ws.max_row
        indexes =[]
        for cols in Prim_cols:
            colindex = find_column_header(cols, ws)
            indexes.append(colindex)
            self.Column_Header.append(cols)
        for cols in Sec_cols:
            colindex = find_column_header(cols, ws)
            indexes.append(colindex)
            self.Column_Header.append(cols)
        self.number_sec_cols = len(Sec_cols)   
        columns = len(indexes)
        self.yvals = np.arange((self.last_row) * columns, dtype = float)
        self.yvals.shape = (columns, self.last_row)
        i = 0
        for cols in indexes:
            if cols != 0:
                for j in range(2, self.last_row + 1):
                    self.yvals[i, j - 1] =  ws.cell(j, cols).value
            else:
                app.update_status("Data not found for chart animation ..")    
            i += 1
        self.X = []
        self.Xscale = []
        for j in range(2, self.last_row):
            temp = ws.cell(j,2).value
            self.X.append(temp)
        self.lines = [None] * columns
        self.defdic = defaultdict(list)
        self.lowlim = 0.0
        self.highlim = 2.0
    
    def initialize(self):
        
        """Initialize graph.  Called from animate_plot. """       
        self.Loop_Ct = 0
        cols = len(self.yvals)
        self.ax.set_xlim(0, 2.0)
        self.ax.set_title(self.title) 
        self.lines = [None] * cols
        return self.lines
        app.update_status("Close animation window to proceed")

    def update(self, j):
        
        """ Animation update routine.  Called from animate_plot. """
        cols = len(self.yvals)
        Colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        if j == 0: 
            # Reset the graph and continue from beginning
            for i in range(0, cols):
                self.defdic[i] = []
                self.lines[i] = []
            plt.cla()
            self.Xscale = []
            self.lowlim = 0
            self.highlim = 2
            self.ax.set_xlim(self.lowlim, self.highlim)
            self.ax.set_ylim(-10, 60)
        self.Xscale.append(self.X[j])
        if j == 1 and self.Loop_Ct == 0:  
            self.Loop_Ct += 1            
            self.ax.legend(loc = 'upper left')
            if self.number_sec_cols > 0:            
                self.ax2.legend(loc = 'upper right')
        # Scale the X axis in 2 hour increments
        if (self.X[j] // 1.0) == 2.0 + self.lowlim:
            self.lowlim += 2.0
            self.highlim += 2.0
            plt.cla()
            self.ax.set_xlim(self.lowlim, self.highlim)
            self.ax.set_ylim(-10, 60)
        for i in range(0, cols):
            index = i % 8
            colorln = Colors[index]
            self.defdic[i].append(self.yvals[i, j])
            if i >= cols - self.number_sec_cols:
                line, = self.ax2.plot(self.Xscale, self.defdic[i],
                                      label = self.Column_Header[i], 
                                      color = colorln)
                self.lines[i].append(line)   
            else:
                line, = self.ax.plot(self.Xscale, self.defdic[i], 
                                     label = self.Column_Header[i], 
                                     color = colorln)
                self.lines[i].append(line)
        if len(self.Xscale) >= 120:
            self.Xscale.pop(0)
            for i in range(0, cols):        
                self.defdic[i].pop(0)   
                self.lines[i].pop(0)
        return self.lines                   
