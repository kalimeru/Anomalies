# Functions
def anomalyScores(originalDF, reducedDF):
    """ Computes the sq.error comparing 2 dataframes
    :originalDF: original dataframe
    :reducedDF: dataframe to be compared
    :return: error
    """
    loss = np.sum((np.array(originalDF) - np.array(reducedDF))**2, axis=1)
    loss = pd.Series(data=loss, index=originalDF.index)
    loss = (loss - np.min(loss)) / (np.max(loss)-np.min(loss))
    return loss

def notNaN(num):
    """ Return if a number is real (not NaN)
    :num: number
    :return: true/false
    """
    return num == num

def add_anomalies(results, list):
    """
    for each model, count anomalies found and update general results list
    :param results: current indexes with amount of times each index has been found as an anomaly
    :param list: current model anomalies
    :return: updated results list
    """
    for i in list.index:
        results.iloc[i] = results.iloc[i] + 1
    return results

def scatterPlot(xDF, yDF, name, cols, cA, cB):
    tempDF = pd.DataFrame(data=xDF.iloc[:,cols], index=xDF.index)
    tempDF = pd.concat((tempDF, yDF), axis=1, join='inner')
    tempDF.columns = [cA, cB, 'Label']
    sns.lmplot(x=cA, y=cB, hue='Label', data=tempDF, fit_reg=False)
    ax=plt.gca()
    ax.set_title('Separation of observations using '+name)

def intersection(a, b):
    both = len([x for x in list(a.index) if x in list(b.index)])
    print(both/len(list(a.index)))

def scan_intersection(a):
    intersection(a, anomalies_PCA)
    intersection(a, anomalies_ica)
    intersection(a, anomalies_svm)
    intersection(a, anomalies_lof)
    intersection(a, anomalies_dbscan)
    intersection(a, anomalies_knn)
    intersection(a, anomalies_gm)
    intersection(a, anomalies_if)

def find_hierarchy(account, hier):
    """ It looks for the parent id of a specific level
    :param account: string with group account
    :param ifrsn: hierarchy dataframe
    :return parent: node parent id
    """
    if account.isnumeric(): # numbers stored as 0000nnnnn -> search 'nnnnn'
        return hier.loc[(hier['id'] == str(int(account))), "main_parent"].values[0]
    else:
        return hier.loc[(hier['id'] == account), "main_parent"].values[0]


def amount_transform(row):
    """ Return -amount if it's credit
    :param df: bseg row
    :return: amount
    """
    return row['Loc.curr.amount'] if row['D/C'] == 'S' else -row['Loc.curr.amount']

def notNaN(num):
    """ Return if a number is real (not NaN)
    :num: number
    :return: true/false
    """
    return num == num

def isWeekend(date):
    """
    Checks if a given date falled into weekend
    :param date: date
    :return: true/false
    """
    weekno = date.weekday()
    if weekno < 5:
        return False
    else:  # 5 Sat, 6 Sun
        return True

def dialog(user_name, usr02):
    """
    Checks if a user is type "dialog user" in SAP
    :param user_name: user logged
    :param usr02: user master table
    :return: true/false
    """
    if (usr02['User Name'].eq(user_name)).any():
        return (usr02.loc[usr02['User Name'] == user_name, 'User Type'] == "A").item()
    else:
        return False

def batch(user_name, usr02):
    """
    Checks if a user logged is type "System"
    :param user_name: user logged
    :param usr02: user master data
    :return: true/false
    """
    if (usr02['User Name'].eq(user_name)).any():
        return (usr02.loc[usr02['User Name'] == user_name, 'User Type'] == "B").item()
    else:
        return False

def get_core(cocd, entity, t001):
    """
    Checks the main business of a company code through entity master data (CMD)
    :param cocd: SAP company code
    :param entity: company master data (CMD)
    :param t001: SAP company code master data
    :return: core segment for the company code
    """
    company = t001.loc[t001['CoCd']==cocd, 'Co.'].iloc[0]
    if str(company)=="nan":
        return ""
    else:
        core = entity.loc[entity['id']==str(int(company)),'core'].iloc[0]
        if str(core)=="nan":
            return ""
        else:
            return core

def find_parent(account, ifrsn, level=4):
    """ It looks for the parent of a specific level
    :param account: group account
    :param ifrsn: dataframe columns account + direct parent
    :param level: level of parent to be return
    :return parent: node parent id
    """

    curr_row = ifrsn.loc[ifrsn['id']==account]

    if curr_row['level'].values[0] <= level: # account is desired level or less
        return account
    else: # recursive with parent
        return find_parent(curr_row['parent'].values[0],ifrsn, level)


def fill_main_parent(ifrsn, level=4):
    """It searches the new parent for accounts
    :param ifrsn: hierarhcy
    :param level: level for which main parent should be searched
    :return: new hierarchy with additional main_parent column
    """
    print(type(ifrsn))
    # accounts are type 1
    account_rows = ifrsn[ifrsn['type']==1]

    ifrsn['main_parent'] = None

    # loop for each account and find father of level 4
    for i in [account for account in account_rows['id']]:
        ifrsn.loc[ifrsn['id']==i, ['main_parent']] = find_parent(i, ifrsn, 4)

    return ifrsn
