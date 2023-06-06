import libraries
import functions

############# param
path = "C:/Users/DANIEL.PONCEMENDOZA/OneDrive - Zurich Insurance/Data Science/"
path_bkpf = path + "Download/bkpf/"
path_bseg = path + "Download/bseg/"
path_in = path + "files/"
path_out = path + "Download/out/"
ws_prefix = "WS"
level = 4

############# files
file_t001 = path_in + "t001.csv"
file_entity = path_in + "Entity.csv"
hierarchy_file = "C:/Users/DANIEL.PONCEMENDOZA/OneDrive - Zurich Insurance/Data analytics/IFRS4_2022Q4.csv"
file_out_hierarchy = path_in + "IFRS4_2022Q4_main_parent.csv"
file_usr02 = path_in + "usr02.csv"
file_out = path_out + "merge.csv"
file_out_doc_type = path_out + "merge_doc_type.csv"
file_out_core = path_out + "merge_core.csv"
file_out_core_subset = path_out + "merge_subset.csv"
bkpf_out = path_out + "bkpf.csv"
bseg_out = path_out + "bseg.csv"
file_out_core_wo_doc_type = path_out + "merge_core_wo_doc_type.csv"
file_out_core_subset_wo_doc_type = path_out + "merge_subset_wo_doc_type.csv"


# Read hierarchy
ifrsn = pd.read_csv(hierarchy_file, sep=",")

# Fill in parent of certain level
hier = fill_main_parent(ifrsn, level)
print(hier.head())

# Write output
hier.to_csv(file_out_hierarchy, index=False)

# read company code master data
t001 = pd.read_csv(file_t001)
print(t001.head())

# read company master data
entity = pd.read_csv(file_entity)
print(entity.head())

# read user type
usr02 = pd.read_csv(file_usr02)
print(usr02.head())

# get all files in bkpf
list_files = [f for f in listdir(path_bkpf)]

# initialize bkpf
bkpf = pd.read_excel(path_bkpf+list_files[0])

# loop for each bkpf file
print("--------------Reading BKPF------------------")
for i in list_files[1:(len(list_files)+1)]:
    curr = pd.read_excel(path_bkpf+i)
    bkpf = pd.DataFrame(np.concatenate([bkpf.values, curr.values]), columns=bkpf.columns)

# save bkpf
bkpf.to_csv(bkpf_out, index=False)

# get all files in bseg
list_files = [f for f in listdir(path_bseg)]

# initialize bseg
bseg = pd.read_excel(path_bseg+list_files[0])

# loop for each bseg file
print("--------------Reading BSEG------------------")
for i in list_files[1:(len(list_files)+1)]:
    curr = pd.read_excel(path_bseg+i)
    bseg = pd.DataFrame(np.concatenate([bseg.values, curr.values]), columns=bseg.columns)

# save bseg
bseg.to_csv(bseg_out, index=False)

# get unique level nodes
unique_nodes = list(set(hier.loc[notNaN(hier['main_parent']),'main_parent']))

# new dataframe at bkpf level + nodes at columns
merge = bkpf
merge = merge.reindex(merge.columns.tolist() + unique_nodes, axis=1)

# add interface flag
merge['interface'] = merge.apply(lambda x: str(x['Doc. Type']).isnumeric(), axis=1)

# add weekend flag
merge['weekend'] = merge.apply(lambda x: isWeekend(x['Entry Dte']), axis=1)

# add winshuttle flag
merge['winshuttle'] = merge.apply(lambda x: str(x['Ref.key (header) 1']).startswith(ws_prefix), axis=1)

# add session name filled in
merge['session_flag'] = merge.apply(lambda x: not str(x['Session name'])=='nan', axis=1)

# add user type
merge['dialog_user'] = merge.apply(lambda x: dialog(x['User Name'], usr02), axis=1)

# add batch user type
merge['batch_user'] = merge.apply(lambda x: batch(x['User Name'], usr02), axis=1)

# add id
merge['id'] = merge.apply(lambda x: x['CoCd'] + '-' + str(x['DocumentNo']) + '-' + str(x['Year']), axis=1)

# get unique document keys for bseg (ignore line item id)
bseg_docs = bseg.loc[:,['CoCd', 'DocumentNo', 'Year']].drop_duplicates()

# find hierarchy level and transform amount (D/C)
bseg['hierarchy'] = bseg.apply(lambda x: find_hierarchy(str(x['Group Acct']), hier), axis=1)
bseg['amount'] = bseg.apply(lambda x: amount_transform(x), axis=1)

# for each document, aggregate amounts, loop for bseg_key in bseg_docs:
for i in range(bseg_docs.shape[0]):
    bseg_key = bseg_docs.iloc[i,:]
    bseg_curr = bseg.loc[(bseg['CoCd']==bseg_key[0]) & (bseg['DocumentNo']==bseg_key[1]) &
                         (bseg['Year']==bseg_key[2])]

    # fill in merge: find document and assign the sum of amounts in the node column
    for node in [i for i in unique_nodes if i in list(set(bseg_curr['hierarchy']))]:
        merge.loc[(merge['CoCd'] == bseg_key[0]) & (merge['DocumentNo'] == bseg_key[1]) &
                  (merge['Year'] == bseg_key[2]), node] = sum(bseg_curr.loc[bseg['hierarchy'] == node, 'amount'])


# all the non-used hierarchies get NaN -> replace with 0
merge.fillna(0, inplace=True)

# business of company codes in scope
merge['core'] = merge.apply(lambda x: get_core(x['CoCd'], entity, t001), axis=1)

# lag between posting date and entry date
merge['lag'] = merge.apply(lambda x: (x['Entry Dte'] - x['Pstng Date']).days, axis=1)

# is the document reversed?
merge['reversed'] = merge.apply(lambda x: x['Reversal']!=0.0, axis=1)

# get doc type in columns (only char ones, ie, without interfaces)
alpha_doc_types = [i for i in list(set(merge['Doc. Type'])) if str(i)[0].isalpha()]
merge['doc_type'] = merge.apply(lambda x: x['Doc. Type'] if x['Doc. Type'] in alpha_doc_types else "", axis=1)
merge_with_type = pd.get_dummies(merge, columns = ['doc_type'])

# get the column name for the newly created one hot encoded columns
new_type_cols = ['doc_type_'+ i for i in alpha_doc_types]

# Column selection
sel_cols_wo_doc_type = ['interface', 'weekend', 'winshuttle', 'dialog_user', 'session_flag',
            'lag', 'reversed'] + unique_nodes

################### Save clean data ##################
# full data set without one hot encoded doc type
merge.to_csv(file_out, index=False)

# full data set with one hot encoded doc type
merge_with_type.to_csv(file_out_doc_type, index=False)

# retain only filled in business
merge_with_core_wo_doc_type = merge[merge['core'].str.len() > 0]

# full data set with core filled in
merge_with_core_wo_doc_type.to_csv(file_out_core_wo_doc_type, index=False)

# full data set with core filled in with selected columns
merge_with_core_wo_doc_type[sel_cols_wo_doc_type].to_csv(file_out_core_subset_wo_doc_type)

# subset per core value
core_set = list(set(merge_with_core_wo_doc_type['core']))
for i in core_set:
    aux = merge_with_core_wo_doc_type.loc[merge_with_core_wo_doc_type['core']==i, sel_cols_wo_doc_type]
    file_temp = path_out + i + "_items_wo_doc_type.csv"
    aux.to_csv(file_temp, index=False)

