"""
Author: Eric Wu
Date: 10.18.2025
Version: b.2

This version of initialization.py is designed to run in python 3.12.x 

Copyright (C) 2025 Eric Wu. All rights reserved.
"""

import numpy as np 
import pandas as pd
import os
import re
import ast
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) #Ignore future warnings. Code runs as intended on pythong 3.12.x

def read_table(folder_path: str,dbug=False):
    """
    DEPRECIATED, 10.13.2025
    This function reads in the pre-determined attributes formmated in the () tables to enable MM operations and for the dataframes to be created. 

    inpts:
        folder_path (str): A string of the file directory of the folder containing the pre-formmated tables stored as a .csv/.db
    opts:
        csv_data (dict): A dictionary of the read in data
    """
    csv_data = dict()
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'): #only read in .csv
            file_path = os.path.join(folder_path,file_name)
            if dbug: print(f"--- Reading file: {file_name} with pandas ---") #dbug
            try:
                df = pd.read_csv(file_path)
                csv_data[file_name] = df
                if dbug: print(df.head()) #dbug 
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return(csv_data)

def read_table_to_dfs(folder_path: str,dbug=False):
    """
    This function reads in the pre-determined attributes formmated in the () tables to enable MM operations and for the dataframes to be created. 

    inpts:
        folder_path (str): A string of the file directory of the folder containing the pre-formmated tables stored as a .csv/.db
    opts:
        (mm_product_master, mm_manufacture_master, mm_resource_master) (tuple of pd.DataFrame objects): returns the dataframes as a tuple 
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'): #only read in .csv
            file_path = os.path.join(folder_path,file_name)
            if dbug: print(f'--- Reading file: {file_name} with pandas ---') #dbug
            try:
                if re.search(r'product', file_name):
                    mm_product_master = pd.read_csv(file_path)
                elif re.search(r'manufacture', file_name):
                    mm_manufacture_master = pd.read_csv(file_path)
                elif re.search(r'resource', file_name):
                    mm_resource_master = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    return (mm_product_master, mm_manufacture_master, mm_resource_master)

def create_mm_dfs(folder_path: str,dbug=False):
    """
    This function creates the dataframes for MM in accordance with the predefined structures and relational integrity as defined in the data dictionary

    inpts:
        folder_path (str): A string of the file directory of the folder containing the pre-formmated tables stored as a .csv/.db
    opts:
        outputs the 7 Mining + Manufacutring DataFrames
    """

    #assign dataframes that where created in read_table_to_dfs 
    (mm_product_master, mm_manufacture_master, mm_resource_master) = read_table_to_dfs(folder_path)
    mm_resource_master['resource_inputs'] = mm_resource_master['resource_inputs'].apply(ast.literal_eval) #fix read in dtype error

    #assign empty dfs for remaining tables
    #mm_location_master
    location_col_defs = {
        'location_coord': object,
        'domain': object,
        'subdomain': object,
        'location_energy': int,
        'required_personelle_num': int,
        'required_personelle_types': object,
        'manufacture_capacity': int,
        'balance': float
    }
    mm_location_master = pd.DataFrame(location_col_defs, index=[])

    #mm_books
    books_col_def = {
        'location_coord': object,
        'period_s': int,
        'period_e': int,
        'balance_s': float,
        'balance_e': float,
        'period_income': float,
        'sales_tax': float
    }
    mm_books = pd.DataFrame(books_col_def, index=[])

    #mm_order_master
    orders_col_def = {
        'order_id': object,
        'product_id': object,
        'product_quantity': int,
        'order_date': int,
        'ship_to': object,
        'ship_by': int,
        'account_receivable': float,
        'order_status': int
    }
    mm_order_master = pd.DataFrame(orders_col_def, index = [])

    #mm_employee_master
    employee_col_def = {
        'location_coord': object,
        'pid': int,
        'wage': float
    }
    mm_employee_master = pd.DataFrame(employee_col_def, index = [])

    return (mm_product_master, mm_manufacture_master, mm_resource_master, mm_location_master, mm_books, mm_order_master, mm_employee_master)

def calc_dist_wrkf_raw(realmc, 
                       n: int, 
                       resource_threshold: float, 
                       chunks: np.ndarray, 
                       min_dist: int, 
                       min_p_size: int, 
                       min_workers: int):
    """
    This function calculates the euclidian distance between the potential workforce chunks and the raw resource locations.
    
    inpts:
        realmc (class Realm obj): E
        n (int): color channel associated with the resource; color channel is (ore, raw, energy)
        resource_threshold (float): rhe value that the color channel must be greater than 
        chunks (ndarray): centers of the n x n chunks that identify the potential workforce in a chunk
        min_dist (int): the minimum distance a chunk must be to a resource
        min_p_size (int): min potential workers per chunk
        min_workers (int): the minimum number of people required for a location type

    opts:
        test_locations_df (pd.DataFrame): a dataframe containing the chunk center, closest resource location, distance, raw_val, and # of potential workers
    """
    #get closest raw mat location to workforces
    resource_mask = np.argwhere(realmc.isle['raw'][:,:,n] > resource_threshold)
    diffs = chunks[:,np.newaxis,:2] - resource_mask
    distance = np.linalg.norm(diffs, axis=2)
    sorted_ndx = np.argsort(distance,axis=1)
    closestvh = resource_mask[sorted_ndx[:,0]]
    distances = distance.min(axis=1)
    loc_arr = np.column_stack([chunks[:,:2], closestvh, distances, realmc.isle['raw'][closestvh[:,0],closestvh[:,1],n], chunks[:,2]])
    test_locations_df = pd.DataFrame(data = loc_arr, columns=['chunk_v','chunk_h','closest_v','closest_h','distance','resource_val','potential_workers'])

    return test_locations_df.query(f'distance <= {min_dist} and potential_workers < {min_p_size} and potential_workers >= {min_workers}')

def add_sel_locs(mm_location_master: pd.DataFrame,
                 tempdf: pd.DataFrame,
                 subdomain: str,
                 loc_gen_dict: dict) -> pd.DataFrame:
    """
    This function adds features and cleans data so the temperary DataFrame containing possible locations can be concatenated with mm_location_master.T

    inpts:
        mm_location_master (pd.DataFrame): location master dataframe
        tempdf (pd.DataFrame): a dataframe containing the chunk center, closest resource location, distance, raw_val, and # of potential workers
        subdomain (str): the subdomain in question
        loc_gen_dict (dict): formated as:
            {subdomain: (
                (int) number of locations of the subdomain to create,
                (int) population size need for subdomain,
                (int) number of employees at location
                )
            }

    opts:
        mm_location_master (pd.DataFrame): location master dataframe
    """
    #column names for parity with mm_location_master
    cols = ['location_coord', 'domain', 'subdomain', 'location_energy', 'required_personelle_num', 'manufacture_capacity', 'balance']

    #clean tempdf
    tempdf['location_coord'] = tempdf[['closest_v','closest_h']].apply(tuple,axis=1)
    tempdf['domain'] = 'resource' if subdomain != 'manufacturing' else 'manufacturing'
    tempdf['subdomain'] = subdomain
    tempdf['required_personelle_num'] = loc_gen_dict[subdomain][2]
    tempdf['location_energy'] = -1 if subdomain != 'energy' else 999
    tempdf['manufacture_capacity'] = 0 if subdomain != 'manufacturing' else 999
    """
    subdomain: balance ==
    farming: 100000,
    mining: 5000000,
    energy: 750000,
    manufacturing: 1000000
    """
    init_balance_scale = {'farming': 100000, 'mining': 500000, 'energy': 750000, 'manufacturing': 1000000} ### NOTE: May need to adjust initial balances
    tempdf['balance'] = init_balance_scale[subdomain]

    #select the possible locations and add them as locations into mm_location_master
    if len(tempdf) >= loc_gen_dict[subdomain][0]:
        mm_location_master = pd.concat([mm_location_master, tempdf[cols].sample(loc_gen_dict[subdomain][0], replace=False)], ignore_index=True)
    else:
        mm_location_master = pd.concat([mm_location_master, tempdf[cols]], ignore_index=True)

    return mm_location_master

def select_location(realmc, 
                    mm_location_master: pd.DataFrame, 
                    loc_gen_dict: dict, 
                    chunk_size: int = 64, 
                    min_workers: int = 5, 
                    resource_threshold: float = 0.9, 
                    rmin_dist: int = 3,
                    dbug: bool = False):
    """
    This function recieves the realm, location master DataFrame, [the types of locations, number to create, the minimum population size, threshold size] and selects locations for 
    farms, mines, energy, and manufacturing locations
    
    inpts:
        realmc (class Realm obj): E
        mm_location_master (pd.DataFrame): location master dataframe
        loc_gen_dict (dict): formated as:
            {subdomain: (
                (int) number of locations of the subdomain to create,
                (int) population size need for subdomain,
                (int) number of employees at location
                )
            }
        chunk_size (int): size of the n x n chunk to consider for eligible worker population
        min_workers (int): threshold for the minimum num of workers discovered in the n x n chunk
        resource_threshold (float): minimum availability of a raw resource
        rmin_dist (int): minimum distance a location can be from a raw resource
    opts:
        mm_location_master (pd.DataFrame): populated location master DataFrame
    """
    #get viable locations based on population
    cols = ['pid','lastname','death', 'locv', 'loch']
    person_vh = realmc.persondf.query(
        f'({realmc.month} - birth) >= 18*12' \
        'and death < 0 ' \
        'and job not in [0.0,1.0]'
    )[cols] #what does job == 2.0???

    #gather potential workforces' locations
    person_vh = person_vh[['locv','loch']].values

    num_bins = int(2048/chunk_size) ### ADD LOGIC: if 2048 % chunk_size > 0 return an exception

    H, v_edges, h_edges = np.histogram2d(
        person_vh[:,0],
        person_vh[:,1],
        bins=num_bins,
        range=[[0, 2048], [0, 2048]] #isle size
    )
    
    dense_chunk_indices = np.argwhere(H >= min_workers) #identify chuncks that have at least n eligable workers

    """densest_chunk_centers = []
    for v_idx, h_idx in dense_chunk_indices:
        v_center = (v_edges[v_idx] + v_edges[v_idx + 1]) / 2
        h_center = (h_edges[h_idx] + h_edges[h_idx + 1]) / 2
        densest_chunk_centers.append([v_center, h_center,H[v_idx,h_idx]])

    densest_chunk_centers = np.array(densest_chunk_centers)"""

    v_idx, h_idx = dense_chunk_indices[:, 0], dense_chunk_indices[:, 1] #get v and h indices
    v_centers, h_centers = v_edges[v_idx] + chunk_size / 2, h_edges[h_idx] + chunk_size / 2 #find the centers of the chunks
    H_values = H[v_idx, h_idx]  #get H (count of people) within the chunks
    
    densest_chunk_centers = np.stack([v_centers, h_centers, H_values], axis=1) #create n x 3 matrix of (v center, h center, count of people)
    if dbug: print(densest_chunk_centers) #dbug


    #populate location table given workforce and raw resource constraints 
    for subdomain in loc_gen_dict.keys():
        if loc_gen_dict[subdomain][0] > 0: #only run if locations are going to be created
            if re.search('farming',subdomain):
                if dbug: print(subdomain, loc_gen_dict[subdomain])
                tempdf = calc_dist_wrkf_raw(realmc, 1, resource_threshold, densest_chunk_centers, rmin_dist, loc_gen_dict[subdomain][1], loc_gen_dict[subdomain][2])
                mm_location_master = add_sel_locs(mm_location_master, tempdf, subdomain, loc_gen_dict)      

            elif re.search('mining',subdomain):
                if dbug: print(subdomain, loc_gen_dict[subdomain])
                tempdf = calc_dist_wrkf_raw(realmc, 0, resource_threshold, densest_chunk_centers, rmin_dist, loc_gen_dict[subdomain][1], loc_gen_dict[subdomain][2])
                mm_location_master = add_sel_locs(mm_location_master, tempdf, subdomain, loc_gen_dict)      

            elif re.search('energy',subdomain):
                if dbug: print(subdomain, loc_gen_dict[subdomain])
                tempdf = calc_dist_wrkf_raw(realmc, 2, resource_threshold, densest_chunk_centers, rmin_dist, loc_gen_dict[subdomain][1], loc_gen_dict[subdomain][2])
                mm_location_master = add_sel_locs(mm_location_master, tempdf, subdomain, loc_gen_dict)      
                
            elif re.search('manufacturing',subdomain):
                if dbug: print(subdomain, loc_gen_dict[subdomain])
                manufacture_loc = densest_chunk_centers[densest_chunk_centers[:,2]>loc_gen_dict[subdomain][2]]
                tempdf = pd.DataFrame(data=manufacture_loc[:,:2],columns=['closest_v','closest_h']) #NOTE: names were choosen for consistency and ease of reusing a function 
                mm_location_master = add_sel_locs(mm_location_master, tempdf, subdomain, loc_gen_dict)  
                
            else:
                if dbug:
                    print(f'{subdomain} is not a valid subdomain for MM, there may be a typo')

    return mm_location_master

def add_books(realmc, location_master: pd.DataFrame, 
              books: pd.DataFrame, 
              dbug = False) -> pd.DataFrame:
    """
    This function populates the books after locations have been selected and populated.

    inpts: 
        realmc (class Realm obj): E
        location_master (pd.DataFrame): expects the mm_location_master DataFrame
        books (pd.DataFrame): expects the mm_books DataFrame
    
    opts:
        books (pd.DataFrame): the mm_books DataFrame after being populated. 
    """
    tempdf = pd.DataFrame() #temp DataFrame to store attributes
    tempdf['location_coord'] = location_master['location_coord'] #add books for ea location
    tempdf['period_s'] = realmc.month #if dbug else 1 #add period start
    tempdf['period_e'] = realmc.month + 2 #if dbug else tempdf['period_s'] + 2
    tempdf['balance_s'] = location_master['balance'] #add starting balance
    tempdf['balance_e'] = 0.0
    tempdf['period_income'] = 0.0
    tempdf['sales_tax'] = 0.0
    books = pd.concat([books, tempdf], ignore_index=True)
    return books

def fill_employee_master(realmc, 
                         location_master: pd.DataFrame, 
                         employee_master: pd.DataFrame, 
                         chunk_size: int = 64, 
                         dbug = False) -> pd.DataFrame:
    """
    This function selects possible workers and hires them given they are within a chunk_size / 2 radius of the location.

    inpts: 
        realmc (class Realm obj): E

        location_master (pd.DataFrame): expects the mm_location_master DataFrame

        employee_master (pd.DataFrame): expects the mm_employee_master DataFrame

        chunk_size (int): used to calculate the radius of searching for workers == "radius"

    opts:
        employee_master (pc.DataFrame): the mm_employee_master DataFrame after being populated
    """
    radius = np.sqrt(2*(chunk_size/2)**2)

    #query for potential workers
    cols = ['pid','lastname','death', 'locv', 'loch']
    person_vh = realmc.persondf.query(
        f'({realmc.month} - birth) >= 18*12' \
        'and death < 0 ' \
        'and lastname != "baby"' \
        'and job not in [0.0,1.0]'
    )[cols] #what does job == 2.0???

    #get locations and required personelle numbers from location_master
    location_df = location_master.copy()
    location_df[['locv_lm', 'loch_lm']] = pd.DataFrame(location_df['location_coord'].tolist(), index=location_df.index) 

    #assign employees to locations!!
    #cross join so ea location is joined to each potential employee
    temp_df = person_vh.assign(key=1).merge(
            location_df.assign(key=1),
            on='key'
        ).drop('key', axis=1)
    
    #calc eudlidian distance
    temp_df['distance'] = np.sqrt((temp_df['locv_lm'] - temp_df['locv'])**2 + (temp_df['loch_lm'] - temp_df['loch'])**2)
    #temp_df['distance'] = np.linalg.norm(np.array(location_master['location_coord'].to_list())[:,np.newaxis,2] - temp_df[['locv', 'loch']].values, axis=1)

    #check distance is less than radius
    eligible_df = temp_df[temp_df['distance'] <= radius].copy()

    #rank the potential employee by distance
    eligible_df['rank'] = eligible_df.groupby('location_coord')['distance'].rank(method='first')

    #ensure actual number of employees selected == required_personelle_num
    final_assignment_df = eligible_df[
            eligible_df['rank'] <= eligible_df['required_personelle_num']
        ].sort_values(['location_coord', 'rank'])
    
    #for dbug; check distances and ranks
    if dbug: print(final_assignment_df[['location_coord', 'pid', 'distance', 'required_personelle_num', 'rank']])

    #clean data so employee master will be clean
    final_assignment_df = final_assignment_df[['location_coord', 'pid']]
    final_assignment_df['wage'] = realmc.mm_params['employee_pay']

    employee_master = pd.concat([employee_master,final_assignment_df],ignore_index=True)

    # add job and info of employees for persondf
    q1 = realmc.persondf['pid'].isin(employee_master['pid'])
    realmc.persondf.loc[q1, 'job'] = 3.0

    return employee_master

def add_prod_res(location_master: pd.DataFrame, 
                 product_master: pd.DataFrame, 
                 resource_master: pd.DataFrame, 
                 use_inventory=False, 
                 dbug=False) -> tuple:
    """
    This function adds a location_cord to the mm_product_master and resource_master DataFrames, given the initialized locations in mm_location_master
    This function also fills in product_count and resource_count as 0 if the use inventory flag is false, else it will be added and specified at a later date

    inpts:
        location_master (pd.DataFrame): expects the mm_location_master DataFrame
        product_master (pd.DataFrame): expects the mm_product_master DataFrame
        resource_master (pd.DataFrame): expects the mm_resource_master DataFrame

    opts:
        product_master (pd.DataFrame): the mm_product_master DataFrame after having location_coord populated
        resource_master (pd.DataFrame): the mm_resource_master DataFrame after having location_coord populated
    """
    #assign products to manufacturing locations
    #sample location_coords from mm_location_master for subdomains == 'manufacturing' to assign location_coord s to products; form many to one relationship
    sample_manf = location_master.loc[location_master['subdomain']=='manufacturing']['location_coord'].sample(n=len(product_master), replace=True, random_state=999)
    product_master['location_coord'] = sample_manf.values.astype(object)

    #assign farming resources to locations
    sample_farm = location_master.loc[location_master['subdomain']=='farming']['location_coord'].sample(n=len(resource_master.loc[resource_master['resource_inputs'].apply(lambda x: x[0]==0)]), replace=True, random_state=999)
    resource_master.loc[
        resource_master['resource_inputs'].apply(lambda x: x[0] == 0),
        'location_coord'
        ] = sample_farm.values.astype(object)

    #assign mined resources to locations
    sample_mine = location_master.loc[location_master['subdomain']=='mining']['location_coord'].sample(n=len(resource_master.loc[resource_master['resource_inputs'].apply(lambda x: x[0]==1)]), replace=True, random_state=999)
    resource_master.loc[
        resource_master['resource_inputs'].apply(lambda x: x[0] == 1),
        'location_coord'
        ] = sample_mine.values.astype(object)

    if dbug: print(f'Number of farms generated|Farms {len(sample_farm), sample_farm} \n Number of mines generated|Mines{len(sample_mine), sample_mine}')

    inventory = [99999,99999]
    product_master['product_count'] = inventory[0] if use_inventory else 0
    resource_master['resource_count'] = inventory[1] if use_inventory else 0

    return (product_master, resource_master)

def fix_dtype(mm_dfs: dict):
    """
    Fixes datadypes after the DataFrames have been populated

    inpts: 
        mm_dfs (dict): A dictionary of the MM DataFrames; the keys are the names of the DataFrames and the values are the DataFrames.

    opts: 
        mm_dfs (dict): A dictionary of the MM DataFrames; the keys are the names of the DataFrames and the values are the DataFrames.
    """
    df_dtype_dict = {
        'mm_product_master': {
            'product_id': object,
            'manufacture_id': object,
            'manufacture_time': int,
            'manufacture_cost': float,
            'product_count': int,
            'location_coord': object,
        },
        'mm_manufacture_master': {
            'manufacture_id': object,
            'resource_id': object,
            'resource_quantity': int
        },
        'mm_resource_master': {
            'resource_id': object,
            'resource_cost': float,
            'resource_count': float,
            'resource_inputs': object,
            'location_coord': object
        },
        'mm_location_master': {
            'location_coord': object,
            'domain': object,
            'subdomain': object,
            'location_energy': int,
            'required_personelle_num': int,
            'required_personelle_types': object,
            'manufacture_capacity': int,
            'balance': float
        },
        'mm_books': {
            'location_coord': object,
            'period_s': int,
            'period_e': int,
            'balance_s': float,
            'balance_e': float,
            'period_income': float,
            'sales_tax': float
        },
        'mm_order_master': {
            'order_id': object,
            'product_id': object,
            'product_quantity': int,
            'order_date': int,
            'ship_to': object,
            'ship_by': int,
            'account_receivable': float,
            'order_status': int
        },
        'mm_employee_master':{
            'location_coord': object,
            'pid': int,
            'wage': float
        }
    }

    for name, df in mm_dfs.items():
        mm_dfs[name] = df.astype(df_dtype_dict[name])

    return mm_dfs

def initialize_mm(realmc, 
                  folder_path: str, 
                  loc_gen_dict: dict = {'farming': (5,25,10),'mining': (5,25,12),'energy': (0,50,20),'manufacturing': (5,100,50)}):
    """
    This function runs the functions needed to populate the MM data structures and add locations given the Main.Realm object.

    inpts:
        realmc (class Realm obj): E
        folder_path (str): A string of the file directory of the folder containing the pre-formmated tables stored as a .csv/.db
        loc_gen_dict (dict): formated as:
            {subdomain: (
                (int) number of locations of the subdomain to create,
                (int) population size need for subdomain,
                (int) number of employees at location
                )
            }

    opts:
        mm_dfs (dict): A dictionary of the MM DataFrames; the keys are the names of the DataFrames and the values are the DataFrames.
    """

    (mm_product_master, mm_manufacture_master, mm_resource_master, mm_location_master, mm_books, mm_order_master, mm_employee_master) = create_mm_dfs(folder_path, realmc.mm_params['dbug'])
    mm_location_master = select_location(realmc, mm_location_master, loc_gen_dict)
    mm_books = add_books(realmc, mm_location_master, mm_books)
    mm_employee_master = fill_employee_master(realmc, mm_location_master, mm_employee_master)
    (mm_product_master, mm_resource_master) = add_prod_res(mm_location_master, mm_product_master, mm_resource_master, False)
    
    mm_dfs = {
        'mm_product_master': mm_product_master,
        'mm_manufacture_master': mm_manufacture_master,
        'mm_resource_master': mm_resource_master,
        'mm_location_master': mm_location_master, 
        'mm_books': mm_books,
        'mm_order_master': mm_order_master,
        'mm_employee_master': mm_employee_master
    }

    mm_dfs = fix_dtype(mm_dfs)

    return mm_dfs

def quick_look(df_dict: dict, n: int = 5):
    """
    Prints the head of each dataframe, given n the number of records to show
    """
    for (key, values) in df_dict.items():
        print(key)
        print(df_dict[key].dtypes)
        print(df_dict[key].head(n), '\n')

def simulate_orders(realmc, 
                    folder_path: str,
                    stores_orders_df: pd.DataFrame = None,
                    set_state: int = 666,
                    n_orders: int = 150,
                    xbar_order_size: int = 500,
                    mn_mx_ar: list = [0.0, 999999.0]
                    ) -> pd.DataFrame: 
    """
    This function simulates the B2B sales between manufacturers and customers.

    inpts:
        realmc (class Realm obj): E:,
        folder_path (str): A string of the file directory of the folder containing the pre-formatted tables stored as a .csv/.db
        stores_orders_df (pd.DataFrame): the qa_sandbox_orders df
        n_orders (int): the number of orders to randomly generate, 
        set_state (int): the seed for generation
        xbar_order_size (int): the average amount of products per order (the function will generate orders +- 3 std from this value),
        mn_mx_ar (list): the min and max account_recievable value of simulated orders, ###NOTE: Future state

        x the type of products ordered, ###NOTE: Future state

    opts:
        stores_orders_df (pd.DataFrame): the qa_sandbox_orders dataframe
    """
    #set rand number generator
    rng = np.random.default_rng(seed=set_state)

    #base prices used to generate orders from
    prod_data_df = pd.read_csv(folder_path+'/qa_sandbox_orders.csv')

    #initialize stores orders DataFrame if None was passed in 
    if stores_orders_df is None:
        #initialize stores orders DataFrame
        placeholder_data, stores_orders_cols = [['o000000', '', 0, 8200, '', 8201, 0, -1]], ['order_id', 'product_id', 'product_quantity', 'order_date', 'ship_to', 'ship_by', 'account_receivable', 'order_status']
        stores_orders_df = pd.DataFrame(data=placeholder_data, columns=stores_orders_cols)

    tempdf = pd.DataFrame() #temp DataFrame

    #generate order_ids ###NOTE: theres a better way to do this...
    init_order_id = stores_orders_df.iloc[-1]['order_id'].lstrip('o').lstrip('0') #remove 'o' and leading 0's
    if init_order_id == '':
        tempdf['order_id'] = ['o'+str(x).zfill(6) for x in range(1,n_orders+1)]
    else:
        tempdf['order_id'] = ['o'+str(x).zfill(6) for x in range(int(init_order_id)+1,int(init_order_id)+n_orders+1)]

    tempdf['product_id'] = prod_data_df['product_id'].sample(n=n_orders, replace=True, random_state=set_state+1).to_list() #generate product_id
    tempdf['product_quantity'] = np.round(rng.normal(50,10,size = n_orders) * xbar_order_size/50).astype(int) #generate product_quantity
    tempdf['order_date'] = realmc.month
    #tempdf['order_status'] = np.zeros(shape=n_orders, dtype=np.int64) #set order_status == 0; order is placed
    tempdf['order_status'] = pd.merge(tempdf, 
                                      realmc.mm_dfs['mm_product_master'][['product_id', 'manufacture_time']], 
                                      how='left',
                                      on='product_id')['manufacture_time'] #assign order status given the manufacture time of a product
    tempdf['ship_to'] = 'qa_simulate_orders' #set ship_to ###NOTE: add logic here

    #left inner join with product data to get order's ship_by and order cost
    tempview = pd.merge(tempdf,prod_data_df, on='product_id', how='left')[['product_id','manufacture_time','sell price']] #set ship_by ###NOTE: add logic here
    tempdf['ship_by'] = tempview['manufacture_time'] + realmc.month
    tempdf['account_receivable'] = tempdf['product_quantity'] * tempview['sell price'] #set account_receivable

    #remove placeholder data after sucessful generation
    if np.any(stores_orders_df['order_id'] == 'o000000', axis=0) == True: stores_orders_df = stores_orders_df[stores_orders_df['order_id'] != 'o000000'].reset_index(drop=True)

    #concat data
    stores_orders_df = pd.concat([stores_orders_df,tempdf], ignore_index=True)
    return stores_orders_df

def calc_cost_profit(realmc,
                     mm_dfs: dict,
                     qa_sandbox_orders: pd.DataFrame) -> pd.DataFrame:
    """
    This function calculates the COGS and profit generated per order.

    inpts:
        realmc (class Realm obj): E
        mm_dataframes (dict): A dictionary of the MM DataFrames; the keys are the names of the DataFrames and the values are the DataFrames
        qa_sandbox_orders: the qa_sandbox_orders dataframe

    opts: 
        a dataframe of the costs and profits given an order
    """

    ord_to_res = pd.merge(qa_sandbox_orders,
                pd.merge(mm_dfs['mm_product_master'], 
                    pd.merge(mm_dfs['mm_manufacture_master'],
                        mm_dfs['mm_resource_master'],
                        on='resource_id'
                    ),
                    on='manufacture_id' 
                ),
                on='product_id'
            ).groupby(
                ['order_id','account_receivable','order_date','product_id','product_quantity','manufacture_time','ship_by','order_status','manufacture_cost']
            )[
                ['resource_id','resource_quantity','resource_cost']
            ].agg(list).reset_index()
    
    pnl_data = ord_to_res[['account_receivable','product_quantity','manufacture_cost','resource_quantity','resource_cost']].to_numpy()

    variable_costs = np.array([
        np.sum(np.array(q) * np.array(c)) 
        for q, c in zip(pnl_data[:, 3], pnl_data[:, 4])
    ])

    costs = pnl_data[:,1] * (pnl_data[:,2] + variable_costs) #cost = product_quantity * (manufacture_cost + resource_quantity * resource_cost)
    sales_tax = pnl_data[:,0] * 0.06 #sales tax
    order_profit = pnl_data[:,0] - costs #- sales_tax #order_profit = revenue - COGS

    ord_to_res['cost'] = costs
    ord_to_res['sales_tax'] = sales_tax
    ord_to_res['profit'] = order_profit
    return ord_to_res