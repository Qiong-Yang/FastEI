# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 07:45:36 2022

@author: jihon
"""

#pyinstaller --add-data="C:/Users/yang/AppData/Roaming/Python/Python37/site-packages/rdkit_pypi.libs;rdkit_pypi.libs" -F "FastEI.py"
import hnswlib
import json
import numpy as np
import pandas as pd
from platform import system

from rdkit import Chem
from rdkit.Chem import Draw
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QVariant
from PyQt5.QtWidgets import QApplication, QGridLayout, QLabel
from PyQt5.Qt import QThread

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from hnswlib import Index
from gensim.models import Word2Vec
import os
import sys
sys.path.append("../..")
from data_process import spec
from data_process.spec_to_wordvector import spec_to_wordvector

from GUI.ui.FastEI_ import Ui_Form


class MakeFigure(FigureCanvas):
    def __init__(self,width=5, height=5, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(top=0.95,bottom=0.2,left=0.15,right=0.95)
        super(MakeFigure,self).__init__(self.fig) 
        self.axes = self.fig.add_subplot(111)
        self.axes.spines['bottom'].set_linewidth(0.5)
        self.axes.spines['left'].set_linewidth(0.5)
        self.axes.spines['right'].set_linewidth(0.5)
        self.axes.spines['top'].set_linewidth(0.5)
        self.axes.tick_params(labelsize=5)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        
    def PlotSpectrum(self, spectrum, reference = None):
        self.axes.cla()
        mz, abunds = spectrum.peaks.mz, spectrum.peaks.intensities
        abunds /= np.max(abunds)
        self.axes.vlines(mz, ymin=0, ymax=abunds, color='r', lw = 1)
        if reference is not None:
            mz1, abunds1 = reference.peaks.mz, reference.peaks.intensities
            abunds1 /= np.max(abunds1)
            self.axes.vlines(mz1, ymin = 0, ymax = -abunds1, color='b', lw = 1)
        self.axes.axhline(y=0,color='black', lw = 1)
        self.axes.set_xlabel('m/z', fontsize = 5)
        self.axes.set_ylabel('abundance', fontsize = 5)
        self.draw()

        
    def PlotComparsion(self, vector, reference):
        self.axes.cla()
        vector, reference = np.array(vector), np.array(reference)
        baseline = min(np.min(vector), np.min(reference))
        ind = np.argsort(vector)
        self.axes.plot(vector[ind] - baseline, color = 'r', lw = 1, label = 'query')
        self.axes.plot(-(reference[ind] - baseline), color = 'b', lw = 1, label = 'reference')
        self.axes.axhline(y=0,color='black', lw = 1)
        self.axes.set_xlabel('index', fontsize = 5)
        self.axes.set_ylabel('value', fontsize = 5)
        self.draw()
    


class TableModel(QtCore.QAbstractTableModel):
    def __init__(self, data, showAllColumn=False):
        QtCore.QAbstractTableModel.__init__(self)
        self.showAllColumn = showAllColumn
        self._data = data


    def rowCount(self, parent=None):
        return self._data.shape[0]


    def columnCount(self, parent=None):
        return self._data.shape[1]


    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None


    def headerData(self,col,orientation,role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if type(self._data.columns[col]) == tuple:
                return self._data.columns[col][-1]
            else:
                return self._data.columns[col]
        elif orientation == Qt.Vertical and role == Qt.DisplayRole:
            return (self._data.axes[0][col])
        return None


class Thread_LoadDatabase(QThread): 
    _compounds = QtCore.pyqtSignal(pd.DataFrame)
    _database = QtCore.pyqtSignal(list)
    
    def __init__(self, db_path):
        super().__init__()
        self.db_path = db_path

    def run(self):       
        Database = spec.load_database(self.db_path)
        compounds = pd.DataFrame({'CompID': [s[0] for s in Database],
                                 'SMILES': [s[1] for s in Database]})
        self._compounds.emit(compounds)
        self._database.emit(Database)


class Thread_LoadIndex(QThread): 
    _index = QtCore.pyqtSignal(hnswlib.Index)
    
    def __init__(self, spec_path):
        super().__init__()
        self.spec_path = spec_path

    def run(self):       
        spec_bin = Index(space = 'l2', dim = 500)
        spec_bin.load_index(self.spec_path)
        self._index.emit(spec_bin)


class FastEI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self, parent=None): 
        super(FastEI, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("FastEI")
        self.label_2.setPixmap(QtGui.QPixmap("FastEI.png"))
        self.setWindowIcon(QtGui.QIcon("FastEI.ico"))
        self.ProcessBar(0, 'ready')
        
        self.figSpe = MakeFigure(1.8, 1.2, dpi = 200)
        self.figSpe_ntb = NavigationToolbar(self.figSpe, self)
        self.gridlayoutfigSpec = QGridLayout(self.groupBoxSpe)
        self.gridlayoutfigSpec.addWidget(self.figSpe)
        self.gridlayoutfigSpec.addWidget(self.figSpe_ntb)
        
        self.Labelmol = QLabel()
        self.gridlayoutMol = QGridLayout(self.groupBoxMol)
        self.gridlayoutMol.addWidget(self.Labelmol)

        self.pushButtonComp.clicked.connect(self.SetCompound)
        self.pushButtonIndex.clicked.connect(self.SetIndex)
        self.pushButtonMod.clicked.connect(self.SetModel)
        self.pushButtonQue.clicked.connect(self.InputQuery)
        # self.pushButtonOk.clicked.connect(self.RunProgram)
        # self.pushButtonView.clicked.connect(self.ViewResult)
        # self.pushButtonPlot.clicked.connect(self.PlotResult)
        self.listWidgetQue.itemClicked.connect(self.ViewResult)
        self.tableWidgetRes.itemClicked.connect(self.PlotResult)
        
        self.allButtons = [self.pushButtonComp,
                           self.pushButtonIndex,
                           self.pushButtonMod,
                           self.pushButtonQue]
                           # self.pushButtonOk,
                           # self.pushButtonView,
                           # self.pushButtonPlot]
        
        self.QueryList = []
        self.Database = []
        self.SpectrumList = []
        self.VectorList = []
        self.SpectrumDB = None
        self.spectovec = None
        self.compounds = None
        self.spec_bin = None
        self.ResultIndex = None
        self.ResultScore = None
        self.Finished = False

        self.Thread_LoadDatabase = None
        self.Thread_LoadIndex = None
        
        # load default
        self.default_index =  os.path.abspath(os.path.join(os.getcwd()))+'/data/references_index.bin'
        self.textBrowserIndex.setText(self.default_index)
        self.default_database =os.path.abspath(os.path.join(os.getcwd()))+'/data/IN_SILICO_LIBRARY.db'
        self.textBrowserComp.setText(self.default_database)
        self.default_model = os.path.abspath(os.path.join(os.getcwd()))+'/data/references_word2vec.model'
        self.textBrowserMod.setText(self.default_model)
        
        self.ProcessBar(30, 'loading index...')
        for s in self.allButtons:
            s.setEnabled(False)
        self.Thread_LoadDatabase = Thread_LoadDatabase(self.default_database)
        self.Thread_LoadDatabase._compounds.connect(self._SetCompound)
        self.Thread_LoadDatabase._database.connect(self._SetDatabase)
        self.Thread_LoadDatabase.start()
        self.Thread_LoadDatabase.finished.connect(self.Load_default)
        
        
    def Load_default(self):
        self.ProcessBar(50, 'loading database...')
        self.Thread_LoadIndex = Thread_LoadIndex(self.default_index)           
        self.Thread_LoadIndex._index.connect(self._SetIndex)
        self.Thread_LoadIndex.start()
        self.Thread_LoadIndex.finished.connect(self._SetIndexFinished)
        
        
    def WarnMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(Text)
        msg.setWindowTitle("Warning")
        msg.exec_()    
    
    
    def ErrorMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(Text)
        msg.setWindowTitle("Error")
        msg.exec_()
        
        
    def InforMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(Text)
        msg.setWindowTitle("Information")
        msg.exec_()
        
        
    def ProcessBar(self, msg, info):
        self.progressBar.setValue(int(msg))
        self.labelStatus.setText(info)


    def SetCompound(self):
        self.Finished = False
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Load", "","database Files (*.db)", options=options)
        if fileName:
            self.textBrowserComp.setText(fileName)
            self.ProcessBar(30, 'loading database...')
            db_path = self.textBrowserComp.toPlainText()
            # self.SpectrumDB = None # need change
            # self.Database = spec.load_database(db_path)
            # self.ProcessBar(70, 'transforming data...')
            # self.compounds = pd.DataFrame({'CompID': [s[0] for s in self.Database],
            #                                'SMILES': [s[1] for s in self.Database]})
            self.Thread_LoadDatabase = Thread_LoadDatabase(db_path)
            self.Thread_LoadDatabase._compounds.connect(self._SetCompound)
            self.Thread_LoadDatabase._database.connect(self._SetDatabase)
            self.Thread_LoadDatabase.start()
            for s in self.allButtons:
                s.setEnabled(False)
            self.Thread_LoadDatabase.finished.connect(self._SetCompoundFinished)
            
        
    def _SetCompound(self, msg):
        self.compounds = msg
    
    
    def _SetDatabase(self, msg):
        self.Database = msg
    
    
    def _SetCompoundFinished(self):
        for s in self.allButtons:
            s.setEnabled(True)
        self.ProcessBar(100, 'Ready!')
        # print(self.compounds)
            

    def SetIndex(self):
        self.Finished = False
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Load", "","Database Files (*.bin)", options=options)
        if fileName:
            self.ProcessBar(30, 'loading spectrum index...')
            self.textBrowserIndex.setText(fileName)
            spec_path = self.textBrowserIndex.toPlainText()
            self.Thread_LoadIndex = Thread_LoadIndex(spec_path)           
            self.Thread_LoadIndex._index.connect(self._SetIndex)
            self.Thread_LoadIndex.start()
            for s in self.allButtons:
                s.setEnabled(False)
            self.Thread_LoadIndex.finished.connect(self._SetIndexFinished)
        else:
            self.ErrorMsg('Invalid spectrum database, please check!')
            
            
    def _SetIndex(self, msg):
        self.spec_bin = msg    
    
    
    def _SetIndexFinished(self):
        for s in self.allButtons:
            s.setEnabled(True)
        self.ProcessBar(100, 'Finished!')            


    def SetModel(self):
        self.Finished = False
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"Load", "","Model Files (*.model)", options=options)
        if fileName:
            self.textBrowserMod.setText(fileName)


    def InputQuery(self):
        self.Finished = False
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"Load", "","CSV Files (*.csv);;TXT Files (*.txt)", options=options)
        
        self.QueryList = []
        self.SpectrumList = []
        self.listWidgetQue.clear()
        
        if len(fileNames) == 0:
            pass
        else:
            self.QueryList, self.SpectrumList = self.ReadMSFiles(fileNames)
            if len(self.QueryList) != len(fileNames):
                self.WarnMsg('Ignore invalid files')
            for fileName in self.QueryList:
                self.listWidgetQue.addItem(fileName)
        self.RunProgram()


    def RunProgram(self):
        for s in self.allButtons:
            s.setEnabled(False)
        
        self.ProcessBar(30, 'loading model...')
        if len(self.QueryList) == 0:
            self.ErrorMsg('No query files input, please check!')
            self.ProcessBar(0, 'ready')
            for s in self.allButtons:
                s.setEnabled(True)
            return
        
        try:
            model_path = self.textBrowserMod.toPlainText()
            model = Word2Vec.load(model_path)
            self.spectovec = spec_to_wordvector(model=model, intensity_weighting_power=0.5)
        except:
            self.ErrorMsg('Invalid model, please check!')
            return

        try:
            pass
        except:
            self.ErrorMsg('Invalid compound list, please check!')
            return
        
        if 'CompID' not in self.compounds.columns:
            self.ErrorMsg('CompID not in column names of compounds, please check!')
            return
        if 'SMILES' not in self.compounds.columns:
            self.ErrorMsg('SMILES not in column names of compounds, please check!')
            return
        
        try:
            pass
        except:
            self.ErrorMsg('Invalid spectrum database, please check!')
            return
        
        try:
            pass
        except:
            self.SpectrumDB = None
        
        self.ProcessBar(50, 'converting data...')
        word2vectors = []
        spectrums = self.SpectrumList
        for i in range(len(spectrums)):
            spectrum_in = spec.SpectrumDocument(spectrums[i], n_decimals=0)
            vetors = self.spectovec._calculate_embedding(spectrum_in)
            word2vectors.append(vetors)
        self.VectorList = word2vectors
        
        self.ProcessBar(70, 'database searching...')
        xq = np.array(word2vectors).astype('float32')
        kn = min(100, self.spec_bin.element_count)
        I, D = self.spec_bin.knn_query(xq, kn)
        self.ResultIndex = I
        self.ResultScore = D
        self.Finished = True
        self.ProcessBar(100, 'ready')
        self.InforMsg('Finished!')
        
        for s in self.allButtons:
            s.setEnabled(True)


    def ViewResult(self):
        if not self.Finished:
            self.ErrorMsg('Please run program first!')
            return
        
        selectItem = self.listWidgetQue.currentItem()
        if not selectItem:
            self.ErrorMsg('No item is selected!')
            return
        else:
            selectItem = selectItem.text()
        
        wh = self.QueryList.index(selectItem)
        index = self.ResultIndex[wh]
        score = self.ResultScore[wh]
        result = self.compounds.loc[index,:]
        result['Distance'] = score
        result = result.reset_index(drop = True)
        result['Rank'] = 1 + np.arange(len(result))
        self.FillResultWidget(result)


    def PlotResult(self):
        selectItem = self.listWidgetQue.currentItem()
        if not selectItem:
            self.ErrorMsg('No item is selected!')
            return
        else:
            selectItem = selectItem.text()        
        wh = self.QueryList.index(selectItem)
        spectrum = self.SpectrumList[wh]
        self.figSpe.PlotSpectrum(spectrum)
        
        header = [self.tableWidgetRes.horizontalHeaderItem(i).text() for i in range(self.tableWidgetRes.columnCount())]
        try:
            i = self.tableWidgetRes.selectedIndexes()[0].row()
        except:
            return
        j = list(header).index('CompID')
        ref = self.tableWidgetRes.item(i, j).text()
        ref = np.where(self.compounds['CompID'] == ref)[0][0]
        k = list(header).index('SMILES')
        smiles = self.tableWidgetRes.item(i, k).text()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return
        PILmol = Draw.MolToQPixmap(mol)
        self.Labelmol.setPixmap(PILmol)
        
        if self.Database is not None:
            refspec = self.Database[ref]
            mz = np.array(json.loads(refspec[3])).astype('float')
            intensity = np.array(json.loads(refspec[4])).astype('float')
            refspec = spec.Spectrum(mz=mz, intensities=intensity)
        else:
            return
        self.figSpe.PlotSpectrum(spectrum, refspec)


    def FillResultWidget(self, data):
        #self.tableWidgetRes.horizontalHeader.setVisible(False)
        self.tableWidgetRes.setRowCount(data.shape[0])
        self.tableWidgetRes.setColumnCount(data.shape[1])
        self.tableWidgetRes.setHorizontalHeaderLabels(data.columns)
        self.tableWidgetRes.setVerticalHeaderLabels(data.index.astype(str))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if type(data.iloc[i,j]) == np.float64:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(Qt.EditRole, QVariant(float(data.iloc[i,j])))
                else:
                    item = QtWidgets.QTableWidgetItem(str(data.iloc[i,j]))
                self.tableWidgetRes.setItem(i, j, item)


    def ReadMSFiles(self, all_files):
        spectrums_m, valid_files = [], []
        for i in range(len(all_files)):
            try:
                f = all_files[i]                
                data = pd.read_csv(f, header = None)
                M = np.array(data.iloc[:,0])
                I = np.array(data.iloc[:,1])
                I = I / np.max(I)
                keep = np.where(I > 0.001)[0]
                M = M[keep].astype(float)
                I = I[keep].astype(float)
                spectrum = spec.Spectrum(mz=M,intensities=I, metadata={'compound_name': str(all_files[i])})
            except:
                continue
            
            valid_files.append(f)
            spectrums_m.append(spectrum)
        return valid_files, spectrums_m


if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    if system() == 'Darwin':
        import _sysconfigdata__darwin_darwin
        app.setWindowIcon(QtGui.QIcon("FastEI.ico"))
    ui = FastEI()
    ui.show()
    sys.exit(app.exec_())
