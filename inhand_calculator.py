import numpy , pandas 
from pandas import DataFrame as df
import matplotlib as plot 



class inhand_generator:
    def __init__(self,data, columns,len = 3):
        self.table = {}
        #print(self.table)
        self.data =data
        self.table = df(self.table, columns= columns )
        #self.table.columns() = columns 
        print(self.table)
    def data_entry(self):
        print(len(self.table))
        x = len(self.table)
        self.table.loc[x] = self.data 
        l = self.tax_calculator()
        self.table.loc[x, 'tax']= (l[0])
        self.table.loc[x, 'rebate']  = l[1]
        self.table.loc[x, 'pf'] = self.pf_generator()
        self.table.loc[x, 'gratuity']  = self.gratuity_gen()
        self.table.loc[x, 'annual_inhand']  = self.total_payable(self.table.loc[x, 'pf'],self.table.loc[x, 'gratuity'] , self.table.loc[x, 'tax'],self.table.loc[x, 'rebate'] )
        self.table.loc[x, 'mnthly_inhand']  = self.table.loc[x, 'annual_inhand'] //12
        print(self.table)
    """Up to ₹4,00,000: Nil
    ₹4,00,001 - ₹8,00,000: 5%
    ₹8,00,001 - ₹12,00,000: 10%
    ₹12,00,001 - ₹16,00,000: 15%
    ₹16,00,001 - ₹20,00,000: 20%
    ₹20,00,001 - ₹24,00,000: 25%
    Above ₹24,00,000: 30%"""

    """
    Taxable Income 	Income Tax Rate
    Up to ₹2,50,000	Nil
    ₹2,50,001 - ₹5,00,000	5%
    ₹5,00,001 - ₹10,00,000	20%
    Above ₹10,00,000	30%"""

    def tax_calculator(self):
        tax =0
        if self.data["tax"] == "new":
            ctc = self.data["ctc"]
            sal = ctc 
            if 0 < ctc <= 400000:
                tax = 0 
                sal = sal - 400000
            if  sal > 0 and 400001 < ctc <=800000:
                tax = tax + 0.05 * sal
                sal = sal - 400000
            if  sal > 0 and 800001 < ctc <=1200000:
                tax = tax + 0.1 * sal
                sal = sal - 400000                             
            if  sal > 0 and 1200001 < ctc <=1600000:
                tax = tax + 0.15 * sal
                sal = sal - 400000
            if  sal > 0 and 1600001 < ctc <=2000000:
                tax = tax + 0.20 * sal
                sal = sal - 400000
            if  sal > 0 and 2000001 < ctc <=2400000:
                tax = tax + 0.25 * sal
                sal = sal - 400000
            if  sal > 0 and 2400001 < ctc:
                tax = tax + 0.30* sal       
            if ctc < 1275000:
                rebate = tax 
            else: rebate = 0

        elif self.data["tax"] == "old":
            ctc = self.data["ctc"]
            sal = ctc 
            if 0 < ctc <= 250000:
                tax = 0 
                sal = sal - 250000
            if  sal > 0 and 250001 < ctc <=500000:
                tax = 0 
                sal = sal - 250000
            if  sal > 0 and 500001 < ctc <=1000000:
                tax = tax + 0.05 * sal
                sal = sal - 500000                             
            if  sal > 0 and 1000001 < ctc :
                tax = tax + 0.1 * sal
            if ctc < 750000:
                rebate = tax 
            else:rebate =0

        
        return tax , rebate 
  

    def total_payable(self, pf, gratuity,tax,rebate):
        total_payable = self.data["ctc"] -pf * 2 - tax - gratuity +rebate
        return total_payable

    def pf_generator(self):
        base = self.data["base"]
        return base * 0.12
    
    def gratuity_gen(self):
        gratuity = self.data["base"] *0.04 
        return gratuity


columns =["ctc","tax", "country","base","bonus"]

data ={
    "ctc": int(input("enter the total CTC "))
    ,"tax": (input("enter the tas regime old/new")).lower()
    ,"country":(input("enter payroll country ")).lower()
    ,"bonus":(int(input("enter the bonus amount")))
}

data ["base"]= data["ctc"] //2

sal =inhand_generator(data, columns )
sal.data_entry()



