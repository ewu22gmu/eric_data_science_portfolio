"""
Author: Eric Wu
Date: 11.17.2025
Version: b.3

This version of operations.py is designed to run in python 3.12.x 

Copyright (C) 2025 Eric Wu. All rights reserved.
"""
###import dependencies
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) #Ignore future warnings. Code runs as intended on pythong 3.12.x

#Pre Evolve Functions
def check_solvency(realmc) -> bool:
    """
    This function should check whether the location is or is not a going concern, given a threshold that businesses are able to go down to.

    ***PHASE2: different thresholds for different subdomains

    inpts:
        Realmc (class Realm obj): E

    opts:
        Returns True if all locations are solvent;
            False if any location is insolvent given the threshold
    """
    # if any balance is less than the debt threshold, then this function will return False, 
    #   because the number of Trues from the series will be less than the number of locations,
    #   which means a location must have had a lower balance than that of mm_debt_thold
    return (realmc.mm_dfs['mm_location_master']['balance'] < realmc.mm_params['mm_debt_thold']).sum() < len(realmc.mm_dfs['mm_location_master'])

def recieve_orders(realmc, ordersdf: pd.DataFrame):
    """
    This function takes the orders recieved and adds them to mm_order_master

    inpts: 
        Realmc (class Realm obj): E,
        ordersdf (pd.DataFrame): the DataFrame of orders from the stores; should follow the data standards of mm_order_master
    """

    #collect new orders
    realmc.mm_dfs['mm_order_master'] = pd.concat([realmc.mm_dfs['mm_order_master'], 
                                                  ordersdf.loc[ordersdf['order_date'] == realmc.month]],
                                                  ignore_index=True).drop_duplicates()

def check_order_parity(realmc, ordersdf: pd.DataFrame):
    """
    This function checks that all of the orders recieved this month where added to mm_order_master.
        If orders are missing, add them into mm_order_master
    """

    #query out orders of this month
    q1 = ordersdf['order_date'] == realmc.month
    q2 = ordersdf['order_date'] > 0
    orderids = ordersdf.loc[q1&q2, 'order_id']

    #check those oreder ids are in mm order master
    if realmc.mm_dfs['mm_order_master']['order_id'].isin(orderids).sum() != len(orderids):
        #add missing orders
        print("there are missing orders")  
        print(f'Missing orders: {realmc.mm_dfs['mm_order_master'].loc[~realmc.mm_dfs['mm_order_master'].isin(orderids)]}')

def preopsMM(realmc, ordersdf: pd.DataFrame):
    """
    This function houses the processes for MM operations that run before the month is updated, before evolve
    """
    if check_solvency(realmc):
        recieve_orders(realmc, ordersdf)
        check_order_parity(realmc, ordersdf)
    #else:
        #print(f'This location is insolvent {realmc.mm_dfs['mm_location_master'].iloc[np.argwhere(check_solvency(realmc))]['location_coord']} at month {realmc.month}') ###NOTE: Check here; may not work.
    
#Post Evolve Functions
def mm_operations(realmc, ccpdf: pd.DataFrame) -> pd.DataFrame:
    """
    This function should run the production process workflow.
        -Calculate which products were ordered, manufactured, and fufiled:
        -Mark order statuses accordingly.
            -mm_order_master.order_status is defined as {-1: 'order_rejected', 0: 'order_complete', 1: 'order_in_production(1m)',
                2: 'order_in_production(2m)', 3: 'order_in_production(3m)', 4: 'order_in_production(4m)'}
            -any order that is [0,4] is assumed accepted.
    """
    def manufacture(realmc, ccpdf: pd.DataFrame) -> pd.DataFrame:
        """
        Manf orders; changes statuses. Passes orders where costs and profits are recognized into update_books
        """

        #filter out order_ids of completed orders and rejected orders:
        #collect new and incomplete orders
        manf_ids = ccpdf.loc[(ccpdf['order_date'] == realmc.month) | (ccpdf['order_status'] > 0)]['order_id'] 

        #collect the order_id
        q1 = realmc.mm_dfs['mm_order_master']['order_id'].isin(manf_ids) 
        realmc.mm_dfs['mm_order_master'].loc[q1,'order_status'] -= 1 #progress through manufacturing

        q2 = realmc.mm_dfs['mm_order_master']['order_status'] == 0
        manf_orders = realmc.mm_dfs['mm_order_master'].loc[q1&q2]

        #return completed orders and pass them out
        return ccpdf.loc[ccpdf['order_id'].isin(manf_orders['order_id'])]

    def update_books(realmc, fufiled_orders: pd.DataFrame):
        """
        Recognizes costs and profits of orders given the orders that have been fufiled; 
            updates mm_location_master.balances and mm_books as appropriate;
            Starts a new entry in mm_books when realmc.month = 1 + 3*i.
        """
        pnl = fufiled_orders.groupby(['product_id']).agg(
                    cost=('cost', 'sum'),
                    tax=('sales_tax', 'sum'),
                    profit=('profit', 'sum')
                ).reset_index()

        pnl_prod = pd.merge(pnl,
                    realmc.mm_dfs['mm_product_master'],
                    how='inner',
                    on='product_id'
                )

        pnl_location = pd.merge(pnl_prod,
                    realmc.mm_dfs['mm_location_master'],
                    how='left',
                    on='location_coord'
                )

        #first filter out latest book entry
        q1 = realmc.mm_dfs['mm_books']['period_s'] == max(realmc.mm_dfs['mm_books']['period_s']) #get latest book
        latest_book = realmc.mm_dfs['mm_books'].loc[q1]


        pnl_books = pd.merge(pnl_prod,
                    latest_book,
                    how='left', 
                    on='location_coord'
                )
        
        
        #update mm_location.balance
        loc_profits = pnl_location.groupby('location_coord').agg(
                profit=('profit','sum')
            ).reset_index()

        loc_profits_indexed = loc_profits.set_index('location_coord')['profit'] #set_index to allign indexes for adding columns

        aligned_profits = realmc.mm_dfs['mm_location_master'].join(
                loc_profits_indexed,
                on='location_coord',
                how='left'
            ).fillna(0.0)['profit'] #join dfs for adding profits

        """realmc.mm_dfs['mm_location_master'].loc[:, 'balance'] = (
                realmc.mm_dfs['mm_location_master']['balance'] + aligned_profits
            )"""
        
        realmc.mm_dfs['mm_location_master']['balance'] += aligned_profits
        
        #update books
        mm_books = realmc.mm_dfs['mm_books'] #temp storage 
        latest_period = mm_books['period_s'].max() #store latest starting period

        loc_books = pnl_books.groupby('location_coord').agg(
                profit=('profit','sum'),
                salestax=('tax','sum')
            ).reset_index() #get profit and sales tax totals

        temp_df = mm_books[mm_books['period_s'] == latest_period].copy() #create temp view to update latest entries

        temp_df = pd.merge(
                left=temp_df,
                right=loc_books,
                on='location_coord',
                how='left'
            )

        q1 = (mm_books['period_s'] == latest_period) #temp query to update latest acct period

        mm_books.loc[q1, 'period_income'] = (
                mm_books.loc[q1, 'period_income'].fillna(0.0).values + temp_df['profit'].fillna(0.0).values
            ) #update income

        mm_books.loc[q1, 'sales_tax'] = (
                mm_books.loc[q1, 'sales_tax'].fillna(0.0) + temp_df['salestax'].fillna(0.0).values
            ) #update sales tax collected figure

        realmc.mm_dfs['mm_books'] = mm_books #save changes

        #Pay resource locations
        ### pass thorugh monies to raw resource locations
        rscdf = fufiled_orders[['order_id', 'product_id', 'product_quantity', 'resource_id', 'resource_quantity', 'resource_cost']].copy()

        rscdf['r_cost'] = rscdf.apply(
                lambda row: (np.round(np.array(row['product_quantity']) * np.array(row['resource_quantity']) * np.array(row['resource_cost']), 2)).tolist(), 
                axis=1
            ) #calc resource cost

        exploded_df = rscdf[['product_id', 'resource_id', 'r_cost']].explode(['resource_id', 'r_cost']) #explode lists stored in r, c

        res_profit = exploded_df.groupby('resource_id').agg(
                resource_profit=('r_cost', 'sum')
            ).reset_index() #get total resource profits

        res_profit = pd.merge(res_profit,
                realmc.mm_dfs['mm_resource_master'],
                how='left',
                on='resource_id'
            ).groupby('location_coord').agg(
                profit=('resource_profit', 'sum')
            ).reset_index() 

        res_profit_index = res_profit.set_index('location_coord')['profit'] #reset index

        aligned_res_profits = realmc.mm_dfs['mm_location_master'].join(
                res_profit_index,
                on='location_coord',
                how='left'
            ).fillna(0.0)['profit'] #join dfs for adding profits

        """realmc.mm_dfs['mm_location_master'].loc[:, 'balance'] = (
                realmc.mm_dfs['mm_location_master']['balance'] + aligned_res_profits
            ) #add profits to mm_location_master"""
        
        realmc.mm_dfs['mm_location_master']['balance'] += aligned_res_profits

        ### update books
        mm_books = realmc.mm_dfs['mm_books'] #temp storage 
        temp_df = mm_books[mm_books['period_s'] == latest_period].copy() #create temp view to update latest entries

        temp_df = pd.merge(
            left=temp_df,
            right=res_profit,
            on='location_coord',
            how='left')

        mm_books.loc[q1, 'period_income'] = (
                mm_books.loc[q1, 'period_income'].fillna(0.0).values + temp_df['profit'].fillna(0.0).values
        ) #update income

        realmc.mm_dfs['mm_books'] = mm_books # save changes

    fufiled_orders = manufacture(realmc, ccpdf)
    update_books(realmc, fufiled_orders)

def mm_tax(realmc, ccpdf: pd.DataFrame):
    """
    This function checks the month to determine the end of an accounting period;
        if it is the end of an accounting period, then taxes are paid accordingly.
    """
    def check_tax(realmc) -> bool:
        """
        This function checks if realmc.month if the end of an accounting period;
            if realmc.month % 3 == 0, then return true, else false
        """

        if realmc.mm_params['begin_month']+3 == realmc.month:
            realmc.mm_params['begin_month'] = realmc.month #Store begining of next acct quarter
            return True
        
    def pay_tax(realmc, ccpdf: pd.DataFrame):
        """
        Define tax type rate as GLOBAL var.
        This function calculates the sales tax and corperate income tax collected during the accounting period (realmc.accounting_period);
            The funds are to be taken from balance to pay for CIT.
        """
        #get begin and end month of acct period to filter for sales
        end_month = realmc.month
        begin_month = end_month - realmc.mm_params['accounting_period'] + 1
        
        q1 = ccpdf['order_date'] >= begin_month 
        q2 = ccpdf['order_date'] <= end_month
        ccpdf_period = ccpdf.loc[q1&q2] 

        #sales tax
        #temp_salestax = pd.merge(realmc.mm_dfs['mm_product_master'][['product_id', 'location_coord']],ccpdf_period, how='right', on='product_id')
        #salestax = temp_salestax.groupby('location_coord')['sales_tax'].sum().reset_index()   

        #CIT  
        q1_1 = realmc.mm_dfs['mm_books']['period_s'] == max(realmc.mm_dfs['mm_books']['period_s']) #get latest bank record
        incm = realmc.mm_dfs['mm_books'].loc[q1_1]['period_income'].fillna(0.0).values
        cit = np.maximum(0.0, incm * realmc.mm_params['cit_rate'])

        #subtract calculated cit tax
        #subtract from current balance
        realmc.mm_dfs['mm_location_master']['balance'] -= cit
        #subtract from period income
        realmc.mm_dfs['mm_books'].loc[q1_1, 'period_income'] -= cit

        #Add new entries and update balance_e if latest period_e == E.month
        ## balance_e = balance_s + period_income
        mm_books = realmc.mm_dfs['mm_books'] #temp storage 
        latest_period = mm_books['period_s'].max() #store latest starting period
        temp_df = mm_books[mm_books['period_s'] == latest_period].copy() #create temp view to update latest entries

        q1 = (mm_books['period_s'] == latest_period) #temp query to update latest acct period

        mm_books.loc[q1, 'balance_e'] = (
                temp_df['balance_s'].fillna(0) + temp_df['period_income'].fillna(0)
            ) #update balance_e

        realmc.mm_dfs['mm_books'] = mm_books #save changes

        ## populate a new set of location coords
        loc_coord = mm_books['location_coord'].unique()
        new_books = pd.DataFrame({'location_coord': loc_coord})
        new_books['period_s'] = realmc.month + 1
        new_books['period_e'] = new_books['period_s'] + 2
        new_books['balance_s'] = realmc.mm_dfs['mm_books'].loc[q1, 'balance_e'].values
        new_books['balance_e'] = 0.0 #will be calculated later
        new_books['period_income'] = 0.0
        new_books['sales_tax'] = 0.0

        realmc.mm_dfs['mm_books'] = pd.concat([realmc.mm_dfs['mm_books'], new_books], ignore_index=True)

        #pay out CIT and salestax
        #return(cit, salestax)

    if check_tax(realmc):
        pay_tax(realmc, ccpdf)

def mm_hr(realmc, chunk_size: int = 64):
    """
    This function deals with ensuring all employees are current; if they are not, people will be hired to replace them;
        this function also pays employees
    """
    def check_employees(realmc, radius: float) -> pd.DataFrame:
        """
        Ensure all employees are current; returns a list of employees who are no longer employable 
            (dead for now; distance will be considered in the future)
        """
        #left join employee master with realmc.persondf
        employees = pd.merge(
            left=realmc.mm_dfs['mm_employee_master'],
            right=realmc.persondf,
            how='left'
        )

        #filter out the employees who have moved away and are dead
        q1 = employees['death'] < 0 #must be alive
        evh, vh = employees[['locv', 'loch']].values, np.array(employees['location_coord'].tolist())
        d = np.linalg.norm(vh - evh, axis=1)
        q2 = d < radius #must be near location they work at d<46

        return employees.loc[~q1|~q2]

    def hire_employees(realmc, replace: pd.DataFrame, radius: float):
        """
        This function recieves a list of employee pids that need to be replaced;
            this function finds new employees to hire according to the replace_list.
        """
        #get list of current employees 
        employee_pid = realmc.mm_dfs['mm_employee_master']['pid'].tolist()

        #get eligible workers near the loc that needs replacing
        replace = replace[['location_coord', 'pid', 'wage']]        

        cols = ['pid','lastname','death', 'locv', 'loch']
        potential_employee = realmc.persondf.query(
            f'({realmc.month} - birth) >= 18*12' \
            'and death < 0 ' \
            'and job not in [0.0,1.0]' \
            f'and pid not in {employee_pid}'
        )[cols]
        
        #cross join to assess possibilities
        temp_df = potential_employee.assign(key=1).merge(replace.assign(key=1), on='key').drop('key', axis=1)

        #calculate distances
        temp_df['distance'] = np.linalg.norm(np.array(temp_df['location_coord'].tolist()) - temp_df[['locv', 'loch']].values, axis=1)
        
        #get eligable bachlors
        eligible_df = temp_df[temp_df['distance'] < radius].copy()
        eligible_df['rank'] = eligible_df.groupby('location_coord')['distance'].rank(method='first')

        #get most eligable worker
        final_assignment_df = eligible_df[
            eligible_df['rank'] == 1
        ].sort_values(['location_coord', 'rank'])

        #hire and assign new employee
        pid_map = final_assignment_df.set_index('pid_y')['pid_x'] #mappping for old worker : new worker
        new_pids = realmc.mm_dfs['mm_employee_master']['pid'].map(pid_map) #add the new worker's pids into mm_employee_master
        realmc.mm_dfs['mm_employee_master']['pid'] = new_pids.fillna(realmc.mm_dfs['mm_employee_master']['pid']).astype(int) #save changes to df

        #adjust new employee's job 
        q1 = realmc.persondf['pid'].isin(final_assignment_df['pid_x'])
        realmc.persondf.loc[q1,'job'] = 3.0 #employed under MM

        #adjust old employee's job 
        q2 = realmc.persondf['pid'].isin(final_assignment_df['pid_y'])
        realmc.persondf.loc[q2,'job'] = 0.0 #no job

    def pay_employees(realmc):
        """
        This function pays the employees a wage of 40,000 / 12 * (1 + np.random.rand(len(realmc.mm_dfs['mm_employee_master']['pids']))/10) 
        """
        temp_employees = realmc.mm_dfs['mm_employee_master'].copy()
        temp_employees['pay'] = (1 + np.random.rand(len(temp_employees))/10) * realmc.mm_params['employee_pay']/12

        #pay employees
        q1 = realmc.persondf['pid'].isin(temp_employees['pid'])
        #add pay to savings
        aligned_pay = realmc.persondf.merge(
                temp_employees[['pid', 'pay']],
                on='pid',
                how='left'
            )['pay'].fillna(0.0)
        
        realmc.persondf.loc[:, 'savings'] = (
                realmc.persondf['savings'].fillna(0) + aligned_pay
            )


        #need to reflect expenses in location_master and books!
        temp_employees_groupby = temp_employees.groupby(['location_coord'])['pay'].sum().reset_index()
        #subtract from mm_location_master.balance
        merge_temp = pd.merge(realmc.mm_dfs['mm_location_master'], 
                              temp_employees_groupby, 
                              on='location_coord', 
                              how='left')
        merge_temp['pay'] = merge_temp['pay'].fillna(0) #handle location w/o employee expenses
        realmc.mm_dfs['mm_location_master']['balance'] = merge_temp['balance'] - merge_temp['pay']

        #subtract from latest mm_books.period_income
        q1 = realmc.mm_dfs['mm_books']['period_s'] == max(realmc.mm_dfs['mm_books']['period_s']) #get latest bank record
        merge_temp = pd.merge(realmc.mm_dfs['mm_books'].loc[q1], 
                              temp_employees_groupby, 
                              how='left', 
                              on='location_coord')
        #recognize expense in mm_books.period_income
        realmc.mm_dfs['mm_books'].loc[q1, 'period_income'] = (
                realmc.mm_dfs['mm_books'].loc[q1, 'period_income'].fillna(0).values - 
                merge_temp['pay'].fillna(0).values
            )

    radius = np.sqrt(2*(chunk_size/2)**2) #radius that employees must be within to stay employed
    replace = check_employees(realmc, radius)
    if len(replace) > 0: hire_employees(realmc, replace, radius)
    pay_employees(realmc)

def postopsMM(realmc, ccpdf: pd.DataFrame):
    """
    This function houses the processes for MM operations that run after the month is updated, after evolve.
    """
    mm_operations(realmc, ccpdf)
    if check_solvency(realmc):
        mm_hr(realmc)
        if check_solvency(realmc):
            mm_tax(realmc, ccpdf)
    #print(realmc.mm_dfs['mm_location_master'])

#test
if __name__ == '__main__':
    #testing
    print('hello Worlb!')